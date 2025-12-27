import os
import numpy as np
import sys

def parse_poscar(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    scale = float(lines[1].strip())
    
    lattice = []
    for i in range(2, 5):
        lattice.append([float(x) for x in lines[i].split()[:3]])
    lattice = np.array(lattice)
    
    elements = lines[5].split()
    counts = list(map(int, lines[6].split()))
    
    if 'Selective' in lines[7].strip():
        coord_start = 9
    else:
        coord_start = 8
    
    atoms = []
    index = 0
    for elem, count in zip(elements, counts):
        for _ in range(count):
            line = lines[coord_start + index].split()
            pos = np.array([float(line[0]), float(line[1]), float(line[2])])
            cart_pos = np.dot(pos, lattice) * scale
            atoms.append({
                'element': elem,
                'index': index + 1,
                'frac_pos': pos,
                'cart_pos': cart_pos
            })
            index += 1
    
    return {
        'scale': scale,
        'lattice': lattice,
        'elements': elements,
        'counts': counts,
        'atoms': atoms
    }

def find_matching_files():
    poscar_files = []
    for f in os.listdir('.'):
        if f.startswith('POSCAR') or f.startswith('CONTCAR'):
            poscar_files.append(f)
    
    if len(poscar_files) < 2:
        print("error: two structure files needed")
        exit(1)
    
    return poscar_files[:2]

def apply_pbc(delta_frac, lattice):
    for i in range(2):
        if delta_frac[i] > 0.5:
            delta_frac[i] -= 1.0
        elif delta_frac[i] < -0.5:
            delta_frac[i] += 1.0

    delta_cart = np.dot(delta_frac, lattice)
    return delta_cart


def main(i):
    element = input("elements (e.g. Pt, O, H): ").strip().capitalize()
    
    file1, file2 = f'CONTCAR', f'CONTCAR (1)'
    #find_matching_files()
    print(f"compare: {file1} and {file2}")
    
    data1 = parse_poscar(file1)
    data2 = parse_poscar(file2)
    
    if element not in data1['elements']:
        print(f"error: element {element} absence")
        exit(1)
    
    if not np.allclose(data1['lattice'], data2['lattice']):
        print("warning: lattice parameter difference")
    
    matched_atoms = []
    for atom1, atom2 in zip(data1['atoms'], data2['atoms']):
        if atom1['element'] == element:
            delta_frac = atom2['frac_pos'] - atom1['frac_pos']
            
            delta_cart = apply_pbc(delta_frac.copy(), data1['lattice'] * data1['scale'])
            
            delta_norm = np.linalg.norm(delta_cart) 
            
            matched_atoms.append({
                'index': atom1['index'],
                'delta_x': delta_cart[0],
                'delta_y': delta_cart[1],
                'delta_z': delta_cart[2],
                'delta_total': delta_norm,
                'pos1': atom1['cart_pos'],
                'pos2': atom2['cart_pos'],
                'delta_frac': delta_frac  
            })
    
    output_lines = []
    output_lines.append("\n (Coordinates: Å):")
    output_lines.append("Num, X, Y, Z, displacement, 1_position(X,Y,Z), 2_position(X,Y,Z)")
    
    for atom in matched_atoms:
        pos1_str = f"{atom['pos1'][0]:.6f}, {atom['pos1'][1]:.6f}, {atom['pos1'][2]:.6f}"
        pos2_str = f"{atom['pos2'][0]:.6f}, {atom['pos2'][1]:.6f}, {atom['pos2'][2]:.6f}"
        line = (f"{atom['index']:4d}, "
              f"{atom['delta_x']:10.6f}, {atom['delta_y']:10.6f}, {atom['delta_z']:10.6f}, "
              f"{atom['delta_total']:10.6f}, "
              f"{pos1_str}, {pos2_str}")
        output_lines.append(line)
    
    if matched_atoms:
        avg_delta = np.mean([a['delta_total'] for a in matched_atoms])
        max_delta = max([a['delta_total'] for a in matched_atoms])
        output_lines.append(f"\ncollective - element {element}:")
        output_lines.append(f"average: {avg_delta:.6f} Å")
        output_lines.append(f"maxinum: {max_delta:.6f} Å")
        
        output_lines.append("\nNote: periodic correction implied")
    else:
        output_lines.append(f"absence {element} atoms")
    
    output_content = "\n".join(output_lines)
    print(output_content)
    
    with open(f"{i}output.txt", "w", encoding="utf-8") as f:
        f.write(f"compare: {file1} and {file2}\n")
        f.write(output_content)
    


for j in range(1):
    if __name__ == "__main__":
        main(j)