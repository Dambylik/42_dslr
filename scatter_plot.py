# one dot = one student
# x = student's Astronomy score
# y = student's Defense Against the Dark Arts score
# color = student's house

import matplotlib.pyplot as plt
from utils import read_csv_file, parse_csv_data


def extract_pair_by_house(rows, headers, feature_x, feature_y):
    house_index = headers.index("Hogwarts House")
    x_index = headers.index(feature_x)
    y_index = headers.index(feature_y)

    houses = {
        "Gryffindor": {"x": [], "y": []},
        "Ravenclaw": {"x": [], "y": []},
        "Hufflepuff": {"x": [], "y": []},
        "Slytherin": {"x": [], "y": []}
    }

    for row in rows:
        if len(row) <= max(house_index, x_index, y_index):
            continue

        house = row[house_index].strip()
        x_val = row[x_index].strip()
        y_val = row[y_index].strip()

        if house not in houses or x_val == "" or y_val == "":
            continue

        try:
            houses[house]["x"].append(float(x_val))
            houses[house]["y"].append(float(y_val))
        except ValueError:
            continue

    return houses


def plot_scatter(houses_data, feature_x, feature_y):
    plt.figure(figsize=(10, 6))

    for house, values in houses_data.items():
        if not values["x"]:
            continue
        plt.scatter(
            values["x"],
            values["y"],
            alpha=0.6,
            s=15,
            label=house
        )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")
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
        
    feature_x = "Astronomy"
    feature_y = "Defense Against the Dark Arts"  # Correlation r=-1.0 (most similar)
    
    houses_data = extract_pair_by_house(rows, headers, feature_x, feature_y)
    plot_scatter(houses_data, feature_x, feature_y)


if __name__ == "__main__":
    main()
    
   