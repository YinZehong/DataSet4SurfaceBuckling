import numpy as np
import sys
import multiprocessing as mp
import argparse
import time

def count_timesteps(filename):
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                count += 1
    return count

def read_lammps_dump(filename):
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('ITEM: TIMESTEP'):
                timestep = int(f.readline().strip())
                f.readline()  # skip NUMBER OF ATOMS
                natoms = int(f.readline().strip())
                f.readline()  # skip BOX BOUNDS
                box_line1 = f.readline().strip().split()
                box_line2 = f.readline().strip().split()
                box_line3 = f.readline().strip().split()
                xlo_bound, xhi_bound, xy = map(float, box_line1)
                ylo_bound, yhi_bound, xz = map(float, box_line2)
                zlo_bound, zhi_bound, yz = map(float, box_line3)

                xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
                xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
                ylo = ylo_bound - min(0.0, yz)
                yhi = yhi_bound - max(0.0, yz)
                zlo = zlo_bound
                zhi = zhi_bound
                lx = xhi - xlo
                ly = yhi - ylo
                lz = zhi - zlo
                H = np.array([[lx, xy, xz],
                              [0, ly, yz],
                              [0, 0, lz]])
                origin = np.array([xlo, ylo, zlo])

                atoms_header = f.readline().strip()
                parts = atoms_header.split()
                if parts[0] == 'ITEM:' and parts[1] == 'ATOMS':
                    columns = parts[2:]
                else:
                    columns = ['id', 'type', 'x', 'y', 'z']
                    print(f"Warning: Unexpected ATOMS header, assuming columns: {columns}", file=sys.stderr)

                try:
                    id_idx = columns.index('id')
                    type_idx = columns.index('type')
                except ValueError:
                    id_idx = 0
                    type_idx = 1
                    print("Warning: 'id' or 'type' not found in columns, using first two columns.", file=sys.stderr)

                if 'x' in columns and 'y' in columns and 'z' in columns:
                    x_idx = columns.index('x')
                    y_idx = columns.index('y')
                    z_idx = columns.index('z')
                    use_scaled = False
                elif 'xs' in columns and 'ys' in columns and 'zs' in columns:
                    x_idx = columns.index('xs')
                    y_idx = columns.index('ys')
                    z_idx = columns.index('zs')
                    use_scaled = True
                else:
                    x_idx, y_idx, z_idx = 2, 3, 4
                    use_scaled = False
                    print("Warning: No coordinate columns (x,y,z or xs,ys,zs) found, assuming first three data columns are x,y,z.", file=sys.stderr)

                atoms = []
                for _ in range(natoms):
                    line = f.readline().strip()
                    if not line:
                        continue
                    atom_parts = line.split()
                    if len(atom_parts) <= max(id_idx, type_idx, x_idx, y_idx, z_idx):
                        continue  
                    atom_id = int(atom_parts[id_idx])
                    atom_type = int(atom_parts[type_idx])
                    if use_scaled:
                        xs = float(atom_parts[x_idx])
                        ys = float(atom_parts[y_idx])
                        zs = float(atom_parts[z_idx])
                        scaled = np.array([xs, ys, zs])
                        cart = origin + H @ scaled
                        x, y, z = cart
                    else:
                        x = float(atom_parts[x_idx])
                        y = float(atom_parts[y_idx])
                        z = float(atom_parts[z_idx])
                    atoms.append((atom_id, atom_type, x, y, z))

                real_box_params = (xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz)
                yield timestep, real_box_params, atoms

def read_lammps_data_simple(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    box_params = None
    atoms = []
    reading_atoms = False
    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0  

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if 'xlo xhi' in line:
            parts = line.split()
            xlo = float(parts[0])
            xhi = float(parts[1])
        elif 'ylo yhi' in line:
            parts = line.split()
            ylo = float(parts[0])
            yhi = float(parts[1])
        elif 'zlo zhi' in line:
            parts = line.split()
            zlo = float(parts[0])
            zhi = float(parts[1])
        elif 'xy xz yz' in line:
            parts = line.split()
            xy = float(parts[0])
            xz = float(parts[1])
            yz = float(parts[2])
            box_params = (xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz)
        elif line.startswith('Atoms'):
            reading_atoms = True
            continue
        elif reading_atoms:
            if not line or not line[0].isdigit():
                reading_atoms = False
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            if len(parts) == 5:  # atomic: id type x y z
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            elif len(parts) == 7:  # full: id mol type charge x y z
                atom_id = int(parts[0])
                atom_type = int(parts[2])
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
            else:
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x = float(parts[-3])
                y = float(parts[-2])
                z = float(parts[-1])
            atoms.append((atom_id, atom_type, x, y, z))

    if box_params is None and all(v is not None for v in [xlo, xhi, ylo, yhi, zlo, zhi]):
        box_params = (xlo, xhi, 0.0, ylo, yhi, 0.0, zlo, zhi, 0.0)
    elif box_params is None:
        raise ValueError("Could not parse box parameters from data file")

    return box_params, atoms

def cluster_pt_by_z(pt_atoms, cluster_thick=1.0):
    if not pt_atoms:
        return [], []
    pt_atoms.sort(key=lambda x: x[1])
    layers = []  
    current_layer_ids = [pt_atoms[0][0]]
    current_z_min = pt_atoms[0][1]
    current_z_max = pt_atoms[0][1]

    for atom_id, z in pt_atoms[1:]:
        if z - current_z_max <= cluster_thick:
            current_layer_ids.append(atom_id)
            current_z_max = z
        else:
            layers.append((current_z_min, current_z_max, current_layer_ids))
            current_layer_ids = [atom_id]
            current_z_min = z
            current_z_max = z
    layers.append((current_z_min, current_z_max, current_layer_ids))

    layers_info = []
    for i, (zmin, zmax, ids) in enumerate(layers):
        layers_info.append((i+1, len(ids), zmin, zmax))

    top_pt_ids = layers[-1][2]

    return layers_info, top_pt_ids

def get_top_pt_ids_from_data_simple(data_file, pt_type, cluster_thick=1.0):

    _, atoms = read_lammps_data_simple(data_file)
    pt_atoms = [(atom[0], atom[4]) for atom in atoms if atom[1] == pt_type]
    if not pt_atoms:
        raise ValueError(f"No Pt atoms (type {pt_type}) found in data file")

    layers_info, top_pt_ids = cluster_pt_by_z(pt_atoms, cluster_thick)

    print(f"Total layers (clustered by z with thickness {cluster_thick}): {len(layers_info)}", file=sys.stderr)
    for idx, n, zmin, zmax in layers_info:
        print(f"Layer {idx}: {n} atoms, z range [{zmin:.3f}, {zmax:.3f}]", file=sys.stderr)

    return top_pt_ids

def get_top_pt_ids_from_first_frame(filename, pt_type, cluster_thick=1.0):

    dump_gen = read_lammps_dump(filename)
    try:
        timestep, box_params, atoms = next(dump_gen)
    except StopIteration:
        raise ValueError("Dump file contains no timestep")
    pt_atoms = [(atom[0], atom[4]) for atom in atoms if atom[1] == pt_type]  # atom[4] is z
    if not pt_atoms:
        raise ValueError(f"No Pt atoms (type {pt_type}) found in the first frame")
    layers_info, top_pt_ids = cluster_pt_by_z(pt_atoms, cluster_thick)
    print(f"Total layers (clustered by z with thickness {cluster_thick}) from first frame: {len(layers_info)}", file=sys.stderr)
    for idx, n, zmin, zmax in layers_info:
        print(f"Layer {idx}: {n} atoms, z range [{zmin:.3f}, {zmax:.3f}]", file=sys.stderr)
    return top_pt_ids

def compute_min_distance(pt, o_atoms, box_params):
    xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz = box_params
    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo
    H = np.array([[lx, xy, xz],
                  [0, ly, yz],
                  [0, 0, lz]])
    origin = np.array([xlo, ylo, zlo])

    pt_frac = np.linalg.solve(H, pt - origin)
    min_dist_sq = float('inf')

    for (_, ox, oy, oz) in o_atoms:
        o_frac = np.linalg.solve(H, np.array([ox, oy, oz]) - origin)
        delta_frac = pt_frac - o_frac
        delta_frac -= np.round(delta_frac)
        delta_cart = H @ delta_frac
        dist_sq = np.dot(delta_cart, delta_cart)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return np.sqrt(min_dist_sq)

def process_frame(frame_data, top_pt_ids, pt_type, o_type, verbose):
    timestep, box_params, atoms = frame_data
    pt_dict = {}
    o_atoms = []
    for atom in atoms:
        if atom[1] == pt_type:
            pt_dict[atom[0]] = (atom[2], atom[3], atom[4])
        elif atom[1] == o_type:
            o_atoms.append((atom[0], atom[2], atom[3], atom[4]))

    top_pt = []
    for pid in top_pt_ids:
        if pid in pt_dict:
            x, y, z = pt_dict[pid]
            top_pt.append((pid, x, y, z))
        else:
            pass

    if not top_pt:
        if verbose:
            return f"{timestep}\t\t\t\t\t\t0.0\t0.0\t0.0\t0.0"
        else:
            return f"{timestep}\t0.0\t0.0\t0.0\t0.0"

    top_pt_ids_present = [pid for pid, _, _, _ in top_pt]
    ptoccu_ids, ptoccu_dists, ptoccu_z = [], [], []
    ptunoccu_ids, ptunoccu_dists, ptunoccu_z = [], [], []

    for pid, px, py, pz in top_pt:
        dist = compute_min_distance(np.array([px, py, pz]), o_atoms, box_params)
        if dist < 2.8:
            ptoccu_ids.append(pid)
            ptoccu_dists.append(dist)
            ptoccu_z.append(pz)
        else:
            ptunoccu_ids.append(pid)
            ptunoccu_dists.append(dist)
            ptunoccu_z.append(pz)

    avg_z_occ = np.mean(ptoccu_z) if ptoccu_z else 0.0
    std_z_occ = np.std(ptoccu_z, ddof=0) if ptoccu_z else 0.0
    avg_z_unocc = np.mean(ptunoccu_z) if ptunoccu_z else 0.0
    std_z_unocc = np.std(ptunoccu_z, ddof=0) if ptunoccu_z else 0.0

    if verbose:
        top_str = ','.join(map(str, top_pt_ids_present))
        occ_ids_str = ','.join(map(str, ptoccu_ids)) if ptoccu_ids else ''
        occ_dist_str = ','.join(f"{d:.6f}" for d in ptoccu_dists) if ptoccu_dists else ''
        unocc_ids_str = ','.join(map(str, ptunoccu_ids)) if ptunoccu_ids else ''
        unocc_dist_str = ','.join(f"{d:.6f}" for d in ptunoccu_dists) if ptunoccu_dists else ''
        return f"{timestep}\t{top_str}\t{occ_ids_str}\t{occ_dist_str}\t{unocc_ids_str}\t{unocc_dist_str}\t{avg_z_occ:.6f}\t{std_z_occ:.6f}\t{avg_z_unocc:.6f}\t{std_z_unocc:.6f}"
    else:
        return f"{timestep}\t{avg_z_occ:.6f}\t{std_z_occ:.6f}\t{avg_z_unocc:.6f}\t{std_z_unocc:.6f}"

def process_chunk(args):
    filename, start_idx, end_idx, top_pt_ids, pt_type, o_type, verbose, shared_counter, lock = args
    results = []
    current_idx = 0
    
    for frame_data in read_lammps_dump(filename):
        current_idx += 1
        if current_idx < start_idx:
            continue
        if current_idx > end_idx:
            break
        
        result_line = process_frame(frame_data, top_pt_ids, pt_type, o_type, verbose)
        results.append(result_line)
        
        with lock:
            shared_counter.value += 1
            
    return results

def main_serial(input_file, output_file, pt_type, o_type, h_type, verbose, top_pt_ids):
    total_frames = count_timesteps(input_file)
    print(f"Total timesteps to process: {total_frames}", file=sys.stderr)
    with open(output_file, 'w') as out:
        if verbose:
            out.write("# Timestep\tTop_Pt_IDs\tPtoccu_IDs\tPtoccu_distances\tPtunoccu_IDs\tPtunoccu_distances\tPtoccu_avg_z\tPtoccu_std_z\tPtunoccu_avg_z\tPtunoccu_std_z\n")
        else:
            out.write("# Timestep\tPtoccu_avg_z\tPtoccu_std_z\tPtunoccu_avg_z\tPtunoccu_std_z\n")
        processed = 0
        for frame_data in read_lammps_dump(input_file):
            processed += 1
            print(f"\rProcessing timestep {processed}/{total_frames} ({processed/total_frames*100:.1f}%)", file=sys.stderr, end='')
            out_line = process_frame(frame_data, top_pt_ids, pt_type, o_type, verbose)
            out.write(out_line + '\n')
        print("\nProcessing complete.", file=sys.stderr)

def main_parallel_chunked(input_file, output_file, pt_type, o_type, h_type, verbose, top_pt_ids, nprocs):
    total_frames = count_timesteps(input_file)
    print(f"Total timesteps to process: {total_frames}", file=sys.stderr)

    manager = mp.Manager()
    shared_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    frames_per_proc = total_frames // nprocs
    chunks = []
    for i in range(nprocs):
        start_idx = i * frames_per_proc + 1
        end_idx = (i + 1) * frames_per_proc if i < nprocs - 1 else total_frames
        chunks.append((input_file, start_idx, end_idx, top_pt_ids, pt_type, o_type, verbose, shared_counter, lock))
        print(f"Process {i+1}: timesteps {start_idx} to {end_idx}", file=sys.stderr)
    
    print(f"Starting parallel processing with {nprocs} processes...", file=sys.stderr)
    pool = mp.Pool(processes=nprocs)
    async_result = pool.map_async(process_chunk, chunks)
    
    while not async_result.ready():
        with lock:
            processed = shared_counter.value
        percentage = processed / total_frames * 100
        print(f"\rProcessing timestep {processed}/{total_frames} ({percentage:.1f}%)", file=sys.stderr, end='')
        time.sleep(0.5)
    
    chunk_results = async_result.get()
    pool.close()
    pool.join()
    
    print(f"\rProcessing timestep {total_frames}/{total_frames} (100.0%)", file=sys.stderr)
    print("\nProcessing complete.", file=sys.stderr)
    
    all_results = []
    for chunk in chunk_results:
        all_results.extend(chunk)
    
    with open(output_file, 'w') as out:
        if verbose:
            out.write("# Timestep\tTop_Pt_IDs\tPtoccu_IDs\tPtoccu_distances\tPtunoccu_IDs\tPtunoccu_distances\tPtoccu_avg_z\tPtoccu_std_z\tPtunoccu_avg_z\tPtunoccu_std_z\n")
        else:
            out.write("# Timestep\tPtoccu_avg_z\tPtoccu_std_z\tPtunoccu_avg_z\tPtunoccu_std_z\n")
        for line in all_results:
            out.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LAMMPS dump for Pt layer buckling.")
    parser.add_argument("input_file", help="LAMMPS dump file")
    parser.add_argument("output_file", help="Output text file")
    parser.add_argument("elem_names", nargs=3, help="Element names for types 1,2,3 (e.g., O H Pt)")
    parser.add_argument("--data", help="Optional LAMMPS data file to define top layer Pt atoms")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (include IDs and distances)")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Number of parallel processes (default 1)")
    args = parser.parse_args()

    elem_names = args.elem_names
    type_map = {name: idx+1 for idx, name in enumerate(elem_names)}
    required = {'Pt', 'O', 'H'}
    if not required.issubset(type_map.keys()):
        print(f"Error: The three element names must include Pt, O, and H (in any order). Got: {elem_names}", file=sys.stderr)
        sys.exit(1)

    pt_type = type_map['Pt']
    o_type = type_map['O']
    h_type = type_map['H']
    verbose = args.verbose
    data_file = args.data
    nprocs = args.parallel

    if data_file:
        try:
            top_pt_ids = get_top_pt_ids_from_data_simple(data_file, pt_type)
            print(f"Top layer Pt atoms from data file: {top_pt_ids}", file=sys.stderr)
        except Exception as e:
            print(f"Error reading data file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            top_pt_ids = get_top_pt_ids_from_first_frame(args.input_file, pt_type)
            print(f"Top layer Pt atoms at timestep 0: {top_pt_ids}", file=sys.stderr)
        except Exception as e:
            print(f"Error reading first frame: {e}", file=sys.stderr)
            sys.exit(1)

    if nprocs <= 1:
        main_serial(args.input_file, args.output_file, pt_type, o_type, h_type, verbose, top_pt_ids)
    else:
        main_parallel_chunked(args.input_file, args.output_file, pt_type, o_type, h_type, verbose, top_pt_ids, nprocs)