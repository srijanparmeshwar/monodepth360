def read_xyz(filename):
    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines()[1:]]

    x = []
    y = []
    z = []
    r = []
    g = []
    b = []
    for line in lines:
        values = [float(value) for value in line.split(" ")]
        x.append(values[0])
        y.append(values[1])
        z.append(values[2])
        r.append(values[3])
        g.append(values[4])
        b.append(values[5])

    return x, y, z, r, g, b