from utils import read_csv_file, parse_csv_data
import matplotlib.pyplot as plt


def extract_course_by_house(rows, headers, course_name):
    house_index = headers.index("Hogwarts House")
    course_index = headers.index(course_name)

    houses = {
        "Gryffindor": [],
        "Ravenclaw": [],
        "Hufflepuff": [],
        "Slytherin": []
    }

    for row in rows:
        if len(row) <= max(house_index, course_index):
            continue

        house = row[house_index].strip()
        grade = row[course_index].strip()

        if house not in houses or grade == "":
            continue

        try:
            houses[house].append(float(grade))
        except ValueError:
            continue

    return houses


def plot_histogram(houses_data, course_name):
    plt.figure(figsize=(10, 6))

    for house, grades in houses_data.items():
        if not grades:
            continue
        plt.hist(grades, bins=20, alpha=0.6, label=house)

    plt.title(f"Score distribution for {course_name}")
    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    # Use default dataset path
    file_path = "datasets/dataset_train.csv"
    
    lines = read_csv_file(file_path)
    headers, rows = parse_csv_data(lines)
    
    # Get all course names (skip non-course columns)
    skip_columns = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday"]
    courses = [h for h in headers if h not in skip_columns]
    
    # Extract data for each course by house
    print("Course data by house:")
    print("-" * 50)
    
    course_name = "Care of Magical Creatures"  # Most homogeneous distribution
    houses_data = extract_course_by_house(rows, headers, course_name)
    plot_histogram(houses_data, course_name)


if __name__ == "__main__":
    main()