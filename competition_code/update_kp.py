import argparse

def update_kp(file_path, new_kp):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if 'Kp: float = ' in line:
            updated_lines.append(f'        Kp: float = {new_kp}\n')
        else:
            updated_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update Kp value in submissions.py')
    parser.add_argument('file_path', type=str, help='Path to submissions.py file')
    parser.add_argument('new_kp', type=float, help='New Kp value')

    args = parser.parse_args()

    update_kp(args.file_path, args.new_kp)
    print(f'Updated Kp value to {args.new_kp} in {args.file_path}')