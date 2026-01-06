import sys

def main():
    """This is the main function of describe.py"""
    
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_name>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    
    if len(lines) == 0:
        print("The dataset is empty.")
        sys.exit(1)
    header_line = lines[0].strip() #removes \n
    data_lines = lines[1:]
    headers = header_line.split(",")

    rows = [] #rows is list[list[str]], each inner list = one student
    for line in data_lines:
        line = line.strip()
        if line == "":
            continue
        row = line.split(",")
        rows.append(row)
    print("Columns:", headers)
    print("Number of rows:", len(rows))


if __name__ == "__main__":
    main()