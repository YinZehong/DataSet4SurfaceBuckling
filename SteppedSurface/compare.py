import argparse
from itertools import product
from pathlib import Path

import numpy as np


ELEMENT = "Pt"
OCCUPIED_BY_ELEMENT = "O"
OCCUPIED_DISTANCE_CUTOFF = 2.8
PT_O_CORE_CUTOFF = 2.45
PT_O_LONG_BOND_SEPARATION = 0.25
PT_O_MIN_OUTWARD_COMPONENT = 0.15
HYDROGEN_ELEMENT = "H"
OH_BOND_CUTOFF = 1.2
PT_NEIGHBOR_CUTOFF = 3.5
TERRACE_COORDINATION = 9
BOTTOM_SEED_COUNT = 12
BOTTOM_LAYER_GAP = 0.5
TOP_LAYER_GAP = 0.5
TOP_LAYER_TOL = 0.08
DEFAULT_TOP_LAYER_COUNT = 18
TERRACE_EXCLUDE_LOW_C_COUNT = 3
TERRACE_EXCLUDE_HIGH_C_COUNT = 3


MASS_ELEMENT_REFERENCES = {
    "H": 1.00794,
    "O": 15.9994,
    "Pt": 195.078,
}


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("zero-length vector")
    return vector / norm


def atom_spatial_label(atom):
    x, y, z = atom["cart_pos"]
    return f"source_global={atom['index']} cart=({x:.8f},{y:.8f},{z:.8f})"


def parse_poscar(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    lattice *= scale

    elements = lines[5].split()
    counts = list(map(int, lines[6].split()))

    mode_line = 7
    selective = lines[mode_line].lower().startswith("s")
    if selective:
        mode_line += 1
    coord_mode = lines[mode_line].lower()
    coord_start = mode_line + 1
    is_direct = coord_mode.startswith("d")
    is_cartesian = coord_mode.startswith(("c", "k"))
    if not (is_direct or is_cartesian):
        raise ValueError(f"unknown coordinate mode in {filename}: {lines[mode_line]}")

    atoms = []
    global_index = 0
    element_indices = {element: 0 for element in elements}
    inv_lattice = np.linalg.inv(lattice)

    for element, count in zip(elements, counts):
        for _ in range(count):
            raw = np.array([float(x) for x in lines[coord_start + global_index].split()[:3]])
            if is_direct:
                frac_pos = raw
                cart_pos = np.dot(frac_pos, lattice)
            else:
                cart_pos = raw * scale
                frac_pos = np.dot(cart_pos, inv_lattice)

            element_indices[element] += 1
            atoms.append(
                {
                    "element": element,
                    "index": global_index + 1,
                    "element_index": element_indices[element],
                    "frac_pos": frac_pos,
                    "cart_pos": cart_pos,
                }
            )
            global_index += 1

    return {
        "format": "vasp",
        "scale": scale,
        "lattice": lattice,
        "elements": elements,
        "counts": counts,
        "atoms": atoms,
    }


def infer_element_from_mass(mass):
    element, _ = min(
        MASS_ELEMENT_REFERENCES.items(),
        key=lambda item: abs(float(mass) - item[1]),
    )
    return element


def parse_lammps_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0
    masses = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo, xhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo, yhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo, zhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
            xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])

    if any(value is None for value in (xlo, xhi, ylo, yhi, zlo, zhi)):
        raise ValueError(f"could not parse LAMMPS box bounds from {filename}")

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo
    lattice = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ]
    )
    origin = np.array([xlo, ylo, zlo])
    inv_lattice = np.linalg.inv(lattice)

    section = None
    atoms_header = ""
    atoms_raw = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        keyword = stripped.split()[0]
        if keyword in {"Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"}:
            section = keyword
            atoms_header = stripped if keyword == "Atoms" else atoms_header
            continue

        if section == "Masses":
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                atom_type = int(parts[0])
                masses[atom_type] = float(parts[1])
        elif section == "Atoms":
            parts = stripped.split()
            if len(parts) < 5 or not parts[0].lstrip("+-").isdigit():
                continue
            atoms_raw.append(parts)

    if not atoms_raw:
        raise ValueError(f"could not parse Atoms section from {filename}")

    type_to_element = {
        atom_type: infer_element_from_mass(mass)
        for atom_type, mass in masses.items()
    }
    if not type_to_element:
        raise ValueError(f"could not parse Masses section from {filename}")

    atoms_style = atoms_header.split("#", 1)[1].strip().lower() if "#" in atoms_header else ""
    atoms = []
    element_indices = {}
    type_counts = {}

    atoms_raw.sort(key=lambda parts: int(parts[0]))
    for parts in atoms_raw:
        atom_id = int(parts[0])
        if atoms_style.startswith("full") and len(parts) >= 7:
            atom_type = int(parts[2])
            coord_start = 4
        elif len(parts) >= 7:
            # Common molecular/full fallback: id mol type ... x y z
            atom_type = int(parts[2])
            coord_start = len(parts) - 6 if len(parts) >= 10 else len(parts) - 3
        else:
            # atomic: id type x y z
            atom_type = int(parts[1])
            coord_start = 2

        element = type_to_element.get(atom_type, f"Type{atom_type}")
        cart_pos = np.array([float(x) for x in parts[coord_start : coord_start + 3]])
        frac_pos = np.dot(cart_pos - origin, inv_lattice)

        element_indices[element] = element_indices.get(element, 0) + 1
        type_counts[element] = type_counts.get(element, 0) + 1
        atoms.append(
            {
                "element": element,
                "index": atom_id,
                "element_index": element_indices[element],
                "frac_pos": frac_pos,
                "cart_pos": cart_pos,
            }
        )

    preferred_order = ["O", "H", "Pt"]
    elements = [element for element in preferred_order if element in type_counts]
    elements.extend(element for element in type_counts if element not in elements)
    counts = [type_counts[element] for element in elements]

    return {
        "format": "lammps-data",
        "scale": 1.0,
        "lattice": lattice,
        "origin": origin,
        "elements": elements,
        "counts": counts,
        "atoms": atoms,
    }


def detect_structure_format(filename):
    with open(filename, "r", encoding="utf-8") as f:
        head = [f.readline() for _ in range(20)]
    text = "".join(head).lower()
    if "lammps data file" in text or any(" atoms" in line.lower() for line in head):
        if any("xlo xhi" in line.lower() for line in head):
            return "lammps-data"
    return "vasp"


def parse_structure(filename):
    structure_format = detect_structure_format(filename)
    if structure_format == "lammps-data":
        return parse_lammps_data(filename)
    return parse_poscar(filename)


def element_atoms(data, element):
    return [atom for atom in data["atoms"] if atom["element"] == element]


def write_numbered_poscar(structure, filename, atoms_by_element, labels_by_atom_id):
    elements = [ELEMENT] if atoms_by_element.get(ELEMENT) else []
    elements.extend(
        element
        for element in structure["elements"]
        if element != ELEMENT and atoms_by_element.get(element)
    )
    elements.extend(
        element
        for element in atoms_by_element
        if element not in elements and atoms_by_element[element]
    )
    counts = [len(atoms_by_element[element]) for element in elements]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Numbered structure generated by compare.py; coordinate comments preserve source IDs\n")
        f.write("1.0\n")
        for vector in structure["lattice"]:
            f.write("  " + "  ".join(f"{value:.12f}" for value in vector) + "\n")
        f.write("  " + "  ".join(elements) + "\n")
        f.write("  " + "  ".join(str(count) for count in counts) + "\n")
        f.write("Direct\n")
        for element in elements:
            for atom in atoms_by_element[element]:
                frac = np.mod(atom["frac_pos"], 1.0)
                label = labels_by_atom_id.get(id(atom), "")
                f.write(
                    "  "
                    + "  ".join(f"{value:.12f}" for value in frac)
                    + (f"  # {label}" if label else "")
                    + "\n"
                )


def export_numbered_structures(reference, target, reference_pt, target_pt, matches, prefix):
    reference_groups = {}
    target_groups = {}
    reference_labels = {}
    target_labels = {}

    for atom in reference["atoms"]:
        reference_groups.setdefault(atom["element"], []).append(atom)
    for atom in target["atoms"]:
        target_groups.setdefault(atom["element"], []).append(atom)

    for atoms in reference_groups.values():
        atoms.sort(key=lambda atom: atom["index"])
    for atoms in target_groups.values():
        atoms.sort(key=lambda atom: atom["index"])

    ordered_reference_pt = sorted(reference_pt, key=lambda atom: atom["index"])
    reference_pt_lookup = {id(atom): i for i, atom in enumerate(reference_pt)}
    for ref_atom in ordered_reference_pt:
        ref_i = reference_pt_lookup[id(ref_atom)]
        target_atom = target_pt[matches[ref_i]["target_index"]]
        reference_labels[id(ref_atom)] = (
            f"{atom_spatial_label(ref_atom)} ref_Pt={ref_atom['element_index']} "
            f"mapped_target_Pt={target_atom['element_index']}"
        )
        target_labels[id(target_atom)] = (
            f"{atom_spatial_label(target_atom)} target_Pt={target_atom['element_index']} "
            f"mapped_ref_Pt={ref_atom['element_index']}"
        )

    reference_groups[ELEMENT] = ordered_reference_pt
    target_groups[ELEMENT] = sorted(target_pt, key=lambda atom: atom["index"])

    for element, atoms in reference_groups.items():
        if element == ELEMENT:
            continue
        for atom in atoms:
            reference_labels[id(atom)] = (
                f"{atom_spatial_label(atom)} {element}={atom['element_index']}"
            )
    for element, atoms in target_groups.items():
        if element == ELEMENT:
            continue
        for atom in atoms:
            target_labels[id(atom)] = (
                f"{atom_spatial_label(atom)} {element}={atom['element_index']}"
            )

    reference_path = Path(f"{prefix}_reference_numbered.vasp")
    target_path = Path(f"{prefix}_target_numbered.vasp")
    write_numbered_poscar(reference, reference_path, reference_groups, reference_labels)
    write_numbered_poscar(target, target_path, target_groups, target_labels)
    return reference_path, target_path


def minimum_image_delta(cart_from, cart_to, lattice, periodic_axes=(0, 1)):
    delta_cart = cart_to - cart_from
    delta_frac = np.linalg.solve(lattice.T, delta_cart)
    periodic_axes = tuple(periodic_axes)
    if not periodic_axes:
        return delta_cart

    centered_frac = delta_frac.copy()
    for axis in periodic_axes:
        centered_frac[axis] -= np.round(centered_frac[axis])

    metric = np.dot(lattice, lattice.T)
    off_diagonal = metric - np.diag(np.diag(metric))
    if np.max(np.abs(off_diagonal)) <= 1e-12 * np.max(np.diag(metric)):
        return np.dot(centered_frac, lattice)

    shortest_delta = None
    shortest_squared = np.inf
    # Component-wise rounding is not sufficient for a skewed cell. Search the
    # neighboring images around the rounded fractional displacement and retain
    # the genuinely shortest Cartesian vector.
    for image_offsets in product((-1, 0, 1), repeat=len(periodic_axes)):
        candidate_frac = centered_frac.copy()
        for axis, image_offset in zip(periodic_axes, image_offsets):
            candidate_frac[axis] += image_offset
        candidate_delta = np.dot(candidate_frac, lattice)
        squared = float(np.dot(candidate_delta, candidate_delta))
        if squared < shortest_squared:
            shortest_squared = squared
            shortest_delta = candidate_delta
    return shortest_delta


def min_distance_to_atoms(cart_pos, atoms, lattice, periodic_axes=(0, 1, 2)):
    if not atoms:
        return float("inf")

    distances = []
    for atom in atoms:
        delta = minimum_image_delta(cart_pos, atom["cart_pos"], lattice, periodic_axes)
        distances.append(np.linalg.norm(delta))
    return min(distances)


def nearest_atom(cart_pos, atoms, lattice, periodic_axes=(0, 1, 2)):
    if not atoms:
        return None, float("inf")

    nearest = None
    nearest_distance = float("inf")
    for atom in atoms:
        delta = minimum_image_delta(cart_pos, atom["cart_pos"], lattice, periodic_axes)
        distance = np.linalg.norm(delta)
        if distance < nearest_distance:
            nearest = atom
            nearest_distance = distance
    return nearest, nearest_distance


def bonded_hydrogens(oxygen_atom, hydrogen_atoms, lattice):
    bonded = []
    for hydrogen_atom in hydrogen_atoms:
        delta = minimum_image_delta(
            oxygen_atom["cart_pos"], hydrogen_atom["cart_pos"], lattice, periodic_axes=(0, 1, 2)
        )
        distance = np.linalg.norm(delta)
        if distance < OH_BOND_CUTOFF:
            bonded.append((hydrogen_atom, distance))
    bonded.sort(key=lambda item: item[1])
    return bonded


def adsorbate_label(h_bond_count):
    if h_bond_count == 2:
        return "H2O"
    if h_bond_count == 1:
        return "OH"
    return f"O(H{h_bond_count})"


def fit_plane_normal(points, orient_toward=None):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = unit(vh[-1])
    if orient_toward is not None and np.dot(orient_toward - centroid, normal) < 0:
        normal = -normal
    return normal, centroid


def bottom_plane_normal(atoms):
    positions = np.array([atom["cart_pos"] for atom in atoms])
    seed_count = min(BOTTOM_SEED_COUNT, len(positions))
    seed_indices = np.argsort(positions[:, 2])[:seed_count]
    normal, centroid = fit_plane_normal(positions[seed_indices], positions.mean(axis=0))

    heights = np.dot(positions - centroid, normal)
    order = np.argsort(heights)
    sorted_heights = heights[order]
    gaps = np.diff(sorted_heights)
    layer_end = 0
    for i, gap in enumerate(gaps):
        if gap > BOTTOM_LAYER_GAP:
            layer_end = i
            break
    else:
        layer_end = seed_count - 1

    bottom_indices = order[: layer_end + 1]
    normal, centroid = fit_plane_normal(positions[bottom_indices], positions.mean(axis=0))
    heights = np.dot(positions - centroid, normal)

    return normal, centroid, bottom_indices, heights


def top_layer_indices(heights, top_count):
    order = np.argsort(heights)
    sorted_heights = heights[order]
    gaps = np.diff(sorted_heights)
    min_top_count = min(top_count, len(order))
    large_gap_indices = [
        i
        for i, gap in enumerate(gaps)
        if gap > TOP_LAYER_GAP and len(order) - i - 1 == min_top_count
    ]

    if large_gap_indices:
        layer_start = large_gap_indices[-1] + 1
        return list(order[layer_start:]), sorted_heights[layer_start]

    top_indices = list(order[-min_top_count:])
    top_cutoff = sorted_heights[-min_top_count]
    return top_indices, top_cutoff


def coordination_numbers(atoms, lattice, cutoff):
    positions = np.array([atom["cart_pos"] for atom in atoms])
    counts = []
    for i, position_i in enumerate(positions):
        count = 0
        for j, position_j in enumerate(positions):
            if i == j:
                continue
            delta = minimum_image_delta(position_i, position_j, lattice, periodic_axes=(0, 1, 2))
            if np.linalg.norm(delta) < cutoff:
                count += 1
        counts.append(count)
    return counts


def top_layer_indices_by_coordination(atoms, lattice, heights, cutoff):
    cn = coordination_numbers(atoms, lattice, cutoff)
    bulk_cn = max(cn)
    surface_indices = [i for i, count in enumerate(cn) if count < bulk_cn]
    if not surface_indices:
        raise ValueError("could not identify surface Pt atoms by coordination")

    surface_order = sorted(surface_indices, key=lambda i: heights[i])
    surface_heights = np.array([heights[i] for i in surface_order])
    if len(surface_order) == 1:
        return surface_order, surface_heights[0], cn, bulk_cn

    surface_midpoint = 0.5 * (surface_heights[0] + surface_heights[-1])
    top_indices = [i for i in surface_order if heights[i] > surface_midpoint]
    if not top_indices:
        gaps = np.diff(surface_heights)
        split_at = int(np.argmax(gaps)) + 1
        top_indices = surface_order[split_at:]

    if not top_indices:
        raise ValueError("could not split top and bottom surface Pt atoms by corrected height")

    top_cutoff = min(heights[i] for i in top_indices)
    return top_indices, top_cutoff, cn, bulk_cn


def terrace_layer_indices_by_coordination(top_indices, coordination, terrace_coordination):
    terrace_indices = [i for i in top_indices if coordination[i] == terrace_coordination]
    non_terrace_indices = [i for i in top_indices if coordination[i] != terrace_coordination]
    return terrace_indices, non_terrace_indices


def extreme_layer_indices(top_indices, heights, low_count, high_count):
    by_height = sorted(top_indices, key=lambda i: heights[i])
    low_count = min(low_count, len(by_height))
    high_count = min(high_count, max(0, len(by_height) - low_count))

    low_indices = by_height[:low_count]
    high_indices = by_height[len(by_height) - high_count :] if high_count else []
    return low_indices, high_indices


def terrace_layer_indices(top_indices, heights, exclude_low_count, exclude_high_count):
    low_indices, high_indices = extreme_layer_indices(
        top_indices, heights, exclude_low_count, exclude_high_count
    )
    excluded = set(low_indices + high_indices)
    terrace_indices = [i for i in top_indices if i not in excluded]

    return terrace_indices, low_indices, high_indices


def corrected_axes(lattice, z_axis):
    z_axis = unit(z_axis)
    x_axis = lattice[0] - np.dot(lattice[0], z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = lattice[1] - np.dot(lattice[1], z_axis) * z_axis
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def minimum_cost_assignment(cost_matrix):
    """Hungarian assignment for a rectangular matrix with rows <= columns."""
    cost = np.asarray(cost_matrix, dtype=float)
    row_count, column_count = cost.shape
    if row_count > column_count:
        raise ValueError("assignment requires no more rows than columns")

    u = np.zeros(row_count + 1)
    v = np.zeros(column_count + 1)
    matched_row = np.zeros(column_count + 1, dtype=int)
    previous_column = np.zeros(column_count + 1, dtype=int)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        min_value = np.full(column_count + 1, np.inf)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = cost[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    previous_column[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = previous_column[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = np.full(row_count, -1, dtype=int)
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    return assignment


def periodic_distance_matrix(first_atoms, second_atoms, lattice):
    distances = np.empty((len(first_atoms), len(second_atoms)))
    for i, first_atom in enumerate(first_atoms):
        for j, second_atom in enumerate(second_atoms):
            delta = minimum_image_delta(first_atom["cart_pos"], second_atom["cart_pos"], lattice)
            distances[i, j] = np.linalg.norm(delta)
    return distances


def match_atoms(reference_atoms, target_atoms, lattice):
    if len(reference_atoms) != len(target_atoms):
        raise ValueError("global atom matching requires equal atom counts")
    distances = periodic_distance_matrix(reference_atoms, target_atoms, lattice)
    assignment = minimum_cost_assignment(distances)
    matches = {}
    for i, j in enumerate(assignment):
        delta = minimum_image_delta(
            reference_atoms[i]["cart_pos"], target_atoms[j]["cart_pos"], lattice
        )
        matches[i] = {
            "target_index": int(j),
            "distance": float(distances[i, j]),
            "delta": delta,
        }
    return matches


def assign_oxygen_to_surface_pt(oxygen_atoms, target_pt_atoms, surface_target_indices, lattice, z_axis):
    surface_atoms = [target_pt_atoms[i] for i in surface_target_indices]
    if not oxygen_atoms or not surface_atoms:
        return {}, []

    distances = periodic_distance_matrix(oxygen_atoms, surface_atoms, lattice)
    bonds_by_target_pt = {}
    conflicts = []
    for oxygen_i, oxygen_atom in enumerate(oxygen_atoms):
        order = np.argsort(distances[oxygen_i])
        surface_column = int(order[0])
        distance = float(distances[oxygen_i, surface_column])
        second_distance = float(distances[oxygen_i, order[1]]) if len(order) > 1 else np.inf
        target_pt_i = surface_target_indices[surface_column]
        target_pt_atom = target_pt_atoms[target_pt_i]
        pt_to_o = minimum_image_delta(
            target_pt_atom["cart_pos"],
            oxygen_atom["cart_pos"],
            lattice,
            periodic_axes=(0, 1),
        )
        outward_component = float(np.dot(pt_to_o, z_axis))
        is_core_bond = distance <= PT_O_CORE_CUTOFF
        is_clear_long_bond = (
            distance <= OCCUPIED_DISTANCE_CUTOFF
            and second_distance - distance >= PT_O_LONG_BOND_SEPARATION
        )
        if outward_component < PT_O_MIN_OUTWARD_COMPONENT or not (
            is_core_bond or is_clear_long_bond
        ):
            continue

        bond = {
            "oxygen": oxygen_atom,
            "distance": distance,
            "second_distance": second_distance,
            "outward_component": outward_component,
            "basis": "core" if is_core_bond else "clear-elongated",
        }
        existing = bonds_by_target_pt.get(target_pt_i)
        if existing is None or distance < existing["distance"]:
            if existing is not None:
                conflicts.append(
                    f"Pt{target_pt_atom['element_index']}: kept O{oxygen_atom['element_index']} "
                    f"instead of O{existing['oxygen']['element_index']}"
                )
            bonds_by_target_pt[target_pt_i] = bond
        else:
            conflicts.append(
                f"Pt{target_pt_atom['element_index']}: kept O{existing['oxygen']['element_index']} "
                f"instead of O{oxygen_atom['element_index']}"
            )
    return bonds_by_target_pt, conflicts


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare terrace Pt displacements between two VASP/POSCAR files, "
            "with corrected surface-normal direction and adsorbate classification."
        )
    )
    parser.add_argument("reference_file", help="Reference VASP/POSCAR file, e.g. 332.vasp")
    parser.add_argument("target_file", help="Target VASP/POSCAR file, e.g. 332OH.vasp")
    parser.add_argument(
        "-o",
        "--output",
        default="top_Pt_displacement.txt",
        help="Output text file. Default: top_Pt_displacement.txt",
    )
    parser.add_argument(
        "--top-count",
        type=int,
        default=DEFAULT_TOP_LAYER_COUNT,
        help=(
            "Number of highest Pt atoms used as stepped top-layer candidates when "
            f"--top-method count is used. Default: {DEFAULT_TOP_LAYER_COUNT}"
        ),
    )
    parser.add_argument(
        "--top-method",
        choices=("coordination", "count"),
        default="coordination",
        help="How to identify stepped top-layer candidates. Default: coordination",
    )
    parser.add_argument(
        "--pt-neighbor-cutoff",
        type=float,
        default=PT_NEIGHBOR_CUTOFF,
        help=f"Pt-Pt neighbor cutoff used for coordination analysis. Default: {PT_NEIGHBOR_CUTOFF}",
    )
    parser.add_argument(
        "--terrace-cn",
        type=int,
        default=TERRACE_COORDINATION,
        help=f"Pt coordination number treated as Pt(111)-like terrace. Default: {TERRACE_COORDINATION}",
    )
    parser.add_argument(
        "--exclude-low-c",
        type=int,
        default=TERRACE_EXCLUDE_LOW_C_COUNT,
        help=f"Number of lowest-c Pt atoms excluded from the top layer. Default: {TERRACE_EXCLUDE_LOW_C_COUNT}",
    )
    parser.add_argument(
        "--exclude-high-c",
        type=int,
        default=TERRACE_EXCLUDE_HIGH_C_COUNT,
        help=f"Number of highest-c Pt atoms excluded from the top layer. Default: {TERRACE_EXCLUDE_HIGH_C_COUNT}",
    )
    parser.add_argument(
        "--allow-same-file",
        action="store_true",
        help="Allow comparing a file with itself. By default this is treated as an input mistake.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reference_file = Path(args.reference_file).resolve()
    target_file = Path(args.target_file).resolve()
    output_file = Path(args.output)
    if reference_file == target_file and not args.allow_same_file:
        raise ValueError(
            "reference_file and target_file are the same path. "
            "Use 1_Surf\\in.vasp as reference and 2_OH\\in.vasp as target."
        )
    print(f"compare: {reference_file} -> {target_file}")

    reference = parse_structure(reference_file)
    target = parse_structure(target_file)

    reference_atoms = element_atoms(reference, ELEMENT)
    target_atoms = element_atoms(target, ELEMENT)
    target_occupying_atoms = element_atoms(target, OCCUPIED_BY_ELEMENT)
    target_hydrogen_atoms = element_atoms(target, HYDROGEN_ELEMENT)
    if not reference_atoms:
        raise ValueError(f"{ELEMENT} not found in {reference_file}")
    if len(reference_atoms) != len(target_atoms):
        raise ValueError(
            f"{ELEMENT} atom count differs: {len(reference_atoms)} in {reference_file.name}, "
            f"{len(target_atoms)} in {target_file.name}"
        )

    normal, plane_point, bottom_indices, heights = bottom_plane_normal(reference_atoms)
    x_axis, y_axis, z_axis = corrected_axes(reference["lattice"], normal)
    matches = match_atoms(reference_atoms, target_atoms, reference["lattice"])
    numbered_prefix = output_file.parent / output_file.stem
    numbered_reference_file, numbered_target_file = export_numbered_structures(
        reference,
        target,
        reference_atoms,
        target_atoms,
        matches,
        numbered_prefix,
    )

    coordination = None
    target_coordination = coordination_numbers(
        target_atoms, target["lattice"], args.pt_neighbor_cutoff
    )
    bulk_coordination = None
    non_terrace_by_coordination = []
    target_embedded_excluded = []
    target_step_excluded = []
    low_c_excluded = []
    high_c_excluded = []
    if args.top_method == "coordination":
        top_indices, top_cutoff, coordination, bulk_coordination = top_layer_indices_by_coordination(
            reference_atoms, reference["lattice"], heights, args.pt_neighbor_cutoff
        )
        top_indices.sort(key=lambda i: reference_atoms[i]["element_index"])
        # Surface relaxation can make a normal terrace atom slightly higher
        # than the atoms at the truncated end, so raw height extrema are not a
        # reliable way to locate the two terminal step rows.  Across PBC the
        # low terminal row gains Pt neighbors; the truncated high row loses
        # them.  Identify those boundary rows by the direction of the CN error.
        low_c_excluded = [i for i in top_indices if coordination[i] > args.terrace_cn]
        high_c_excluded = [i for i in top_indices if coordination[i] < args.terrace_cn]
        height_excluded = set(low_c_excluded + high_c_excluded)
        height_filtered_indices = [i for i in top_indices if i not in height_excluded]
        terrace_indices, non_terrace_by_coordination = terrace_layer_indices_by_coordination(
            height_filtered_indices, coordination, args.terrace_cn
        )
    else:
        top_indices, top_cutoff = top_layer_indices(heights, args.top_count)
        top_indices.sort(key=lambda i: reference_atoms[i]["element_index"])
        height_filtered_indices, geometric_low_excluded, geometric_high_excluded = terrace_layer_indices(
            top_indices, heights, args.exclude_low_c, args.exclude_high_c
        )
        coordination = coordination_numbers(
            reference_atoms, reference["lattice"], args.pt_neighbor_cutoff
        )
        bulk_coordination = max(coordination)
        low_c_excluded = list(geometric_low_excluded)
        high_c_excluded = list(geometric_high_excluded)
        for i in height_filtered_indices:
            if coordination[i] > args.terrace_cn:
                low_c_excluded.append(i)
            elif coordination[i] < args.terrace_cn:
                high_c_excluded.append(i)
        terrace_indices, non_terrace_by_coordination = terrace_layer_indices_by_coordination(
            height_filtered_indices, coordination, args.terrace_cn
        )

    reference_cn9_indices = list(terrace_indices)
    terrace_indices = []
    for ref_i in reference_cn9_indices:
        target_i = matches[ref_i]["target_index"]
        target_cn = target_coordination[target_i]
        if target_cn > args.terrace_cn:
            target_embedded_excluded.append(ref_i)
        elif target_cn < args.terrace_cn:
            target_step_excluded.append(ref_i)
        else:
            terrace_indices.append(ref_i)
    terrace_indices.sort(
        key=lambda i: (heights[i], reference_atoms[i]["element_index"])
    )
    terrace_target_indices = [matches[i]["target_index"] for i in terrace_indices]
    surface_target_indices = [matches[i]["target_index"] for i in top_indices]
    pt_o_bonds, pt_o_conflicts = assign_oxygen_to_surface_pt(
        target_occupying_atoms,
        target_atoms,
        surface_target_indices,
        target["lattice"],
        z_axis,
    )
    terrace_target_set = set(terrace_target_indices)
    nonterrace_surface_bonds = [
        (target_i, bond)
        for target_i, bond in pt_o_bonds.items()
        if target_i not in terrace_target_set
    ]

    output_lines = []
    output_lines.append("")
    output_lines.append(f"Reference format: {reference['format']}")
    output_lines.append(f"Target format: {target['format']}")
    output_lines.append(f"Numbered reference VASP: {numbered_reference_file}")
    output_lines.append(f"Numbered target VASP: {numbered_target_file}")
    output_lines.append(
        "Numbered VASP rule: atoms within each element are ordered by original source atom ID; "
        "each coordinate-line comment binds the source ID to its original Cartesian (x,y,z), "
        "and Pt comments separately record the spatially mapped Pt number"
    )
    output_lines.append("Corrected axes:")
    output_lines.append(f"X axis: [{x_axis[0]: .8f}, {x_axis[1]: .8f}, {x_axis[2]: .8f}]")
    output_lines.append(f"Y axis: [{y_axis[0]: .8f}, {y_axis[1]: .8f}, {y_axis[2]: .8f}]")
    output_lines.append(f"Z axis: [{z_axis[0]: .8f}, {z_axis[1]: .8f}, {z_axis[2]: .8f}]")
    output_lines.append(f"Bottom plane Pt atoms used: {len(bottom_indices)}")
    output_lines.append(f"Top layer cutoff along corrected Z: {top_cutoff:.6f} Angstrom")
    output_lines.append(f"Top Pt atoms selected: {len(top_indices)}")
    if args.top_method == "coordination":
        top_coord_counts = {}
        for i in top_indices:
            top_coord_counts[coordination[i]] = top_coord_counts.get(coordination[i], 0) + 1
        coord_summary = ", ".join(
            f"CN{coordination_number}: {count}"
            for coordination_number, count in sorted(top_coord_counts.items())
        )
        output_lines.append(
            f"Top layer method: Pt-Pt coordination < bulk CN {bulk_coordination}, "
            "then keep the upper surface by corrected Z"
        )
        output_lines.append(f"Pt-Pt neighbor cutoff: {args.pt_neighbor_cutoff:.3f} Angstrom")
        output_lines.append(f"Top layer coordination summary: {coord_summary}")
        output_lines.append(
            f"Terrace rule: require Pt coordination CN{args.terrace_cn} in both reference "
            "and mapped target; lower CN is step/edge and higher CN is embedded/step-adjacent"
        )
    else:
        output_lines.append(f"Top layer method: highest corrected-Z Pt count = {args.top_count}")
        output_lines.append(
            f"Terrace rule: exclude the {args.exclude_low_c} lowest-c and "
            f"{args.exclude_high_c} highest-c Pt atoms from the top layer, then require "
            f"CN{args.terrace_cn} in both reference and mapped target"
        )
    output_lines.append(f"Terrace Pt atoms selected: {len(terrace_indices)}")
    output_lines.append(
        "TXT atom-list order: reference corrected-c/surface-normal height from low to high"
    )
    output_lines.append(
        ("Excluded embedded/step-adjacent Pt in reference (CN > terrace): " if args.top_method == "coordination" else "Excluded low-height or high-CN Pt in reference: ")
        + (
            ", ".join(
                f"{reference_atoms[i]['element_index']}"
                f"(ref CN{coordination[i] if coordination is not None else 0}, "
                f"target {target_atoms[matches[i]['target_index']]['element_index']} "
                f"CN{target_coordination[matches[i]['target_index']]}, "
                f"h={heights[i]:.6f})"
                for i in sorted(low_c_excluded, key=lambda idx: heights[idx])
            )
            if low_c_excluded
            else "none"
        )
    )
    output_lines.append(
        ("Excluded step/edge Pt in reference (CN < terrace): " if args.top_method == "coordination" else "Excluded high-height or low-CN Pt in reference: ")
        + (
            ", ".join(
                f"{reference_atoms[i]['element_index']}"
                f"(ref CN{coordination[i] if coordination is not None else 0}, "
                f"target {target_atoms[matches[i]['target_index']]['element_index']} "
                f"CN{target_coordination[matches[i]['target_index']]}, "
                f"h={heights[i]:.6f})"
                for i in sorted(high_c_excluded, key=lambda idx: heights[idx])
            )
            if high_c_excluded
            else "none"
        )
    )
    if args.top_method == "coordination":
        output_lines.append(
            "Other non-terrace-CN Pt after terminal-row exclusion: "
            + (
                ", ".join(
                    f"{reference_atoms[i]['element_index']}(CN{coordination[i]}, "
                    f"h={heights[i]:.6f})"
                    for i in sorted(non_terrace_by_coordination, key=lambda idx: heights[idx])
                )
                if non_terrace_by_coordination
                else "none"
            )
        )
    output_lines.append(
        "Excluded embedded/step-adjacent Pt in target (target CN > terrace): "
        + (
            ", ".join(
                f"{reference_atoms[i]['element_index']}->{target_atoms[matches[i]['target_index']]['element_index']}"
                f"(target CN{target_coordination[matches[i]['target_index']]})"
                for i in target_embedded_excluded
            )
            if target_embedded_excluded
            else "none"
        )
    )
    output_lines.append(
        "Excluded step/edge Pt in target (target CN < terrace): "
        + (
            ", ".join(
                f"{reference_atoms[i]['element_index']}->{target_atoms[matches[i]['target_index']]['element_index']}"
                f"(target CN{target_coordination[matches[i]['target_index']]})"
                for i in target_step_excluded
            )
            if target_step_excluded
            else "none"
        )
    )
    output_lines.append("")
    output_lines.append(
        "Num, TargetNum, Ref_Pt_CN, Target_Pt_CN, Occupancy, Pt_O_distance, next_Pt_distance, bond_basis, occupying_O_num, "
        "O_H_bonds, adsorbate, dX, dY, dZ, displacement, ref_Z, target_Z, "
        "ref_position(X,Y,Z), target_position(X,Y,Z)"
    )

    displacements = []
    occupied = []
    unoccupied = []
    occupied_by_adsorbate = {}
    surface_mappings = []
    unoccupied_vertical_displacements = []
    for ref_i in terrace_indices:
        ref_atom = reference_atoms[ref_i]
        match = matches[ref_i]
        target_atom = target_atoms[match["target_index"]]
        delta_cart = match["delta"]
        delta_local = np.array(
            [
                np.dot(delta_cart, x_axis),
                np.dot(delta_cart, y_axis),
                np.dot(delta_cart, z_axis),
            ]
        )

        ref_height = np.dot(ref_atom["cart_pos"] - plane_point, z_axis)
        target_height = np.dot(target_atom["cart_pos"] - plane_point, z_axis)
        target_pt_i = match["target_index"]
        pt_o_bond = pt_o_bonds.get(target_pt_i)
        is_occupied = pt_o_bond is not None
        if is_occupied:
            occupying_o = pt_o_bond["oxygen"]
            min_o_distance = pt_o_bond["distance"]
            next_pt_distance = pt_o_bond["second_distance"]
            bond_basis = pt_o_bond["basis"]
        else:
            occupying_o, min_o_distance = nearest_atom(
                target_atom["cart_pos"], target_occupying_atoms, target["lattice"]
            )
            next_pt_distance = np.nan
            bond_basis = "-"
        occupancy = "occupied" if is_occupied else "unoccupied"
        h_bonds = bonded_hydrogens(occupying_o, target_hydrogen_atoms, target["lattice"]) if is_occupied else []
        h_bond_count = len(h_bonds)
        adsorbate = adsorbate_label(h_bond_count) if is_occupied else "-"
        occupying_o_num = occupying_o["element_index"] if is_occupied else 0
        displacements.append(np.linalg.norm(delta_cart))

        occupancy_info = {
            "ref_num": ref_atom["element_index"],
            "ref_global_index": ref_atom["index"],
            "pt_coordination": coordination[ref_i] if coordination is not None else 0,
            "target_pt_coordination": target_coordination[match["target_index"]],
            "target_num": target_atom["element_index"],
            "target_global_index": target_atom["index"],
            "occupying_o_num": occupying_o_num,
            "min_o_distance": min_o_distance,
            "next_pt_distance": next_pt_distance,
            "bond_basis": bond_basis,
            "h_bond_count": h_bond_count,
            "adsorbate": adsorbate,
            "vertical_displacement": delta_local[2],
        }
        surface_mappings.append(
            {
                "ref_num": ref_atom["element_index"],
                "ref_global_index": ref_atom["index"],
                "pt_coordination": coordination[ref_i] if coordination is not None else 0,
                "target_pt_coordination": target_coordination[match["target_index"]],
                "target_num": target_atom["element_index"],
                "target_global_index": target_atom["index"],
                "match_distance": match["distance"],
                "occupancy": occupancy,
                "occupying_o_num": occupying_o_num,
                "bond_basis": bond_basis,
                "h_bond_count": h_bond_count,
                "adsorbate": adsorbate,
            }
        )
        if is_occupied:
            occupied.append(occupancy_info)
            occupied_by_adsorbate.setdefault(adsorbate, []).append(occupancy_info)
        else:
            unoccupied.append(occupancy_info)
            unoccupied_vertical_displacements.append(delta_local[2])

        ref_pos = ref_atom["cart_pos"]
        target_pos = target_atom["cart_pos"]
        output_lines.append(
            f"{ref_atom['element_index']:4d}, {target_atom['element_index']:9d}, "
            f"{(coordination[ref_i] if coordination is not None else 0):5d}, "
            f"{target_coordination[match['target_index']]:12d}, "
            f"{occupancy:10s}, {min_o_distance:13.6f}, {next_pt_distance:16.6f}, "
            f"{bond_basis:15s}, {occupying_o_num:15d}, {h_bond_count:9d}, {adsorbate:9s}, "
            f"{delta_local[0]:10.6f}, {delta_local[1]:10.6f}, {delta_local[2]:10.6f}, "
            f"{np.linalg.norm(delta_cart):10.6f}, "
            f"{ref_height:10.6f}, {target_height:10.6f}, "
            f"{ref_pos[0]:.6f}, {ref_pos[1]:.6f}, {ref_pos[2]:.6f}, "
            f"{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}"
        )

    if displacements:
        output_lines.append("")
        output_lines.append(f"Terrace {ELEMENT} mapping ({reference_file.name} -> {target_file.name}):")
        output_lines.append(
            "RefPtNum, RefGlobalIndex, Ref_Pt_CN, TargetPtNum, TargetGlobalIndex, Target_Pt_CN, match_distance, "
            "Occupancy, bond_basis, occupying_O_num, O_H_bonds, adsorbate"
        )
        for mapping in surface_mappings:
            output_lines.append(
                f"{mapping['ref_num']:4d}, {mapping['ref_global_index']:14d}, "
                f"{mapping['pt_coordination']:5d}, "
                f"{mapping['target_num']:11d}, {mapping['target_global_index']:17d}, "
                f"{mapping['target_pt_coordination']:12d}, "
                f"{mapping['match_distance']:14.6f}, {mapping['occupancy']:10s}, "
                f"{mapping['bond_basis']:15s}, {mapping['occupying_o_num']:15d}, "
                f"{mapping['h_bond_count']:9d}, "
                f"{mapping['adsorbate']}"
            )
        output_lines.append("")
        output_lines.append(f"collective - terrace {ELEMENT}:")
        output_lines.append(f"average: {np.mean(displacements):.6f} Angstrom")
        output_lines.append(f"maximum: {np.max(displacements):.6f} Angstrom")
        output_lines.append("")
        output_lines.append(
            f"Occupancy criterion: each O is bonded only to its nearest Pt among all stepped "
            f"top-surface Pt; accept distance <= {PT_O_CORE_CUTOFF:.3f} Angstrom, or a clear "
            f"elongated bond <= {OCCUPIED_DISTANCE_CUTOFF:.3f} Angstrom when the next Pt is at "
            f"least {PT_O_LONG_BOND_SEPARATION:.3f} Angstrom farther; outward normal component "
            f"must be >= {PT_O_MIN_OUTWARD_COMPONENT:.3f} Angstrom"
        )
        output_lines.append(
            "O bonded to excluded non-terrace surface Pt (embedded/step/edge; not counted): "
            + (
                ", ".join(
                    f"O{bond['oxygen']['element_index']}->Pt"
                    f"{target_atoms[target_i]['element_index']}({bond['distance']:.6f})"
                    for target_i, bond in nonterrace_surface_bonds
                )
                if nonterrace_surface_bonds
                else "none"
            )
        )
        output_lines.append(
            "Multiple O competing for one Pt: "
            + ("; ".join(pt_o_conflicts) if pt_o_conflicts else "none")
        )
        output_lines.append(
            f"Adsorbate criterion: O-H distance < {OH_BOND_CUTOFF:.3f} Angstrom; "
            "2 bonded H = H2O, 1 bonded H = OH"
        )
        output_lines.append(
            "Occupied Pt: "
            + (
                ", ".join(
                    f"{item['ref_num']}->{item['target_num']}"
                    f"[O{item['occupying_o_num']},{item['adsorbate']},"
                    f"{item['min_o_distance']:.6f}]"
                    for item in occupied
                )
                if occupied
                else "none"
            )
        )
        output_lines.append(
            "Unoccupied Pt: "
            + (
                ", ".join(
                    f"{item['ref_num']}->{item['target_num']}({item['min_o_distance']:.6f})"
                    for item in unoccupied
                )
                if unoccupied
                else "none"
            )
        )
        output_lines.append("")
        output_lines.append(f"Occupied {ELEMENT} vertical displacements by adsorbate along corrected Z:")
        for label in ("H2O", "OH"):
            items = occupied_by_adsorbate.get(label, [])
            values = [item["vertical_displacement"] for item in items]
            pt_list = (
                ", ".join(
                    f"{item['ref_num']}->{item['target_num']}"
                    f"(O{item['occupying_o_num']}, dZ={item['vertical_displacement']:.6f})"
                    for item in items
                )
                if items
                else "none"
            )
            output_lines.append(f"{label}-occupied Pt: {pt_list}")
            if values:
                output_lines.append(f"{label} average: {np.mean(values):.6f} Angstrom")
                output_lines.append(f"{label} STDEV.P: {np.std(values, ddof=0):.6f} Angstrom")
            else:
                output_lines.append(f"{label} average: none")
                output_lines.append(f"{label} STDEV.P: none")
        other_labels = sorted(label for label in occupied_by_adsorbate if label not in {"H2O", "OH"})
        for label in other_labels:
            items = occupied_by_adsorbate[label]
            values = [item["vertical_displacement"] for item in items]
            output_lines.append(
                f"{label}-occupied Pt: "
                + ", ".join(
                    f"{item['ref_num']}->{item['target_num']}"
                    f"(O{item['occupying_o_num']}, dZ={item['vertical_displacement']:.6f})"
                    for item in items
                )
            )
            output_lines.append(f"{label} average: {np.mean(values):.6f} Angstrom")
            output_lines.append(f"{label} STDEV.P: {np.std(values, ddof=0):.6f} Angstrom")
        output_lines.append("")
        output_lines.append(f"Unoccupied {ELEMENT} vertical displacements along corrected Z:")
        if unoccupied_vertical_displacements:
            output_lines.append(
                ", ".join(f"{value:.6f}" for value in unoccupied_vertical_displacements)
            )
            output_lines.append(
                f"average: {np.mean(unoccupied_vertical_displacements):.6f} Angstrom"
            )
            output_lines.append(
                f"STDEV.P: {np.std(unoccupied_vertical_displacements, ddof=0):.6f} Angstrom"
            )
        else:
            output_lines.append("none")
        output_lines.append("")
        output_lines.append("Note: dZ is projected onto the fitted bottom-plane normal.")
        output_lines.append("Note: periodic correction is applied in lattice a/b directions.")
    else:
        output_lines.append(f"no terrace {ELEMENT} atoms selected")

    output_content = "\n".join(output_lines)
    print(output_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"compare: {reference_file} -> {target_file}\n")
        f.write(output_content)
    print(f"\nwritten: {output_file}")
    print(f"written: {numbered_reference_file}")
    print(f"written: {numbered_target_file}")


if __name__ == "__main__":
    main()
