from utils import read_csv_file, parse_csv_data
from describe import calculate_variance, extract_numerical_columns, calculate_mean, calculate_std
import matplotlib.pyplot as plt
import sys
import os


def normalize_data(rows, headers):
    house_index = headers.index("Hogwarts House")
    houses = {
        "Gryffindor": [],
        "Ravenclaw": [],
        "Hufflepuff": [],
        "Slytherin": []
    }
    
    for row in rows:
        house = row[house_index].strip()
        if house in houses:
            houses[house].append(row)
    var_data = {}
    for house_name, data_list in houses.items():
        numerical_columns = extract_numerical_columns(headers, data_list)
        var_data[house_name] = {}
        for col_name, values in numerical_columns.items():
            sorted_values = sorted(values)
            mean = calculate_mean(sorted_values)
            variance = calculate_variance(sorted_values, mean)
            var_data[house_name][col_name] = variance
    # print(f"Variance: {var_data}")
    course_variances = {}
    first_house = list(var_data.keys())[0]
    course_names = list(var_data[first_house].keys())
    for course in course_names:
        variances_list = []
        for house in var_data:
            v = var_data[house][course]
            variances_list.append(v)
        course_variances[course] = variances_list
    std_res = {}
    for course, val_list in course_variances.items():
        sorted_values = sorted(val_list)
        mean = calculate_mean(sorted_values)
        std = calculate_std(sorted_values, mean)
        std_res[course] = std
    std_res_sorted = dict(sorted(std_res.items(), key=lambda item: item[1]))
    print(f"STD: {std_res_sorted}")
    return std_res_sorted

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
    try: 
        assert len(sys.argv) == 2, "Usage: python histogram.py <dataset_path>"
    except AssertionError as e: 
        print(f"Error: {e}")
        sys.exit(1)
    file_path = sys.argv[1]
    try: 
        if os.path.getsize(file_path) > 0:
            lines = read_csv_file(file_path)
            headers, rows = parse_csv_data(lines)
        else:
            print(f"Error: empty file was provided")
            sys.exit(1)
    except OSError as e:
        print("Error: {e}")
        sys.exit(1)
    std_res_sorted = normalize_data(rows, headers)
    course_name = list(std_res_sorted.keys())[0]
    print(f"first course = {course_name}")
    # course_name = "Care of Magical Creatures"  # Most homogeneous distribution
    houses_data = extract_course_by_house(rows, headers, course_name)
    plot_histogram(houses_data, course_name)


if __name__ == "__main__":
    main()