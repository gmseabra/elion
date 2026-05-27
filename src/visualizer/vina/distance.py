import math

def get_coordinate(label):
    while True:
        try:
            raw = input(f"  {label}: ").strip()
            x, y, z = map(float, raw.replace(",", " ").split())
            return x, y, z
        except ValueError:
            print("  ⚠  Please enter exactly 3 numbers separated by spaces (e.g. 1 2 3)")

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def main():
    print("=" * 40)
    print("   3D Distance Calculator")
    print("=" * 40)
    print("Enter each point as:  x y z  (space-separated)\n")

    p1 = get_coordinate("Point 1 (x y z)")
    p2 = get_coordinate("Point 2 (x y z)")

    dist = euclidean_distance(p1, p2)

    print("\n" + "-" * 40)
    print(f"  Point 1 : ({p1[0]}, {p1[1]}, {p1[2]})")
    print(f"  Point 2 : ({p2[0]}, {p2[1]}, {p2[2]})")
    print(f"  Distance: {dist:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    main()