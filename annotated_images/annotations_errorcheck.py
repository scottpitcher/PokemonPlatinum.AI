import os

def check_annotation_files(annotations_dir):
    files = sorted(os.listdir(annotations_dir))
    errors = []

    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(annotations_dir, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_number, line in enumerate(f, start=1):
                    elements = line.strip().split()
                    if len(elements) != 5:
                        errors.append({
                            'file': file,
                            'line_number': line_number,
                            'content': line.strip(),
                            'num_elements': len(elements)
                        })

    return errors

def main():
    annotations_dir = 'annotated_images/labels'  # Update this path as needed
    errors = check_annotation_files(annotations_dir)

    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors:
            print(f"File: {error['file']}, Line: {error['line_number']}, Elements: {error['num_elements']}")
            print(f"Content: {error['content']}\n")
    else:
        print("No errors found in annotation files.")

if __name__ == "__main__":
    main()
