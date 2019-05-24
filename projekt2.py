import glob, os, sys
import cv2
import numpy as np
from skimage import io
from skimage.util import invert
from skimage.transform import rotate
from skimage.morphology import dilation, binary_dilation, disk, rectangle
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt


def get_rows(maxima, size):
    lines = sorted(maxima)
    halfs = [(lines[i] + lines[i+1])/2 for i in range(len(lines)-1)]
    bounds = []
    for j in range(len(lines)):
        if j == 0:
            bounds.append([0, int(halfs[0])])
        elif j == len(lines) -1:
            bounds.append([int(halfs[-1]), size])
        else:
            bounds.append([int(halfs[j-1]), int(halfs[j])])

    return bounds


def przytnij(rows, cols):
    ig = 0
    id = len(rows) - 1
    jl = 0
    jp = len(cols) - 1

    while rows[ig] >= len(rows) / 40:
        ig +=1
    while rows[id] >= len(rows) / 40:
        id -=1
    while cols[jl] >= len(cols) / 3:
        jl +=1
    while cols[jp] >= len(cols) / 3:
        jp -=1

    return ig, id, jl, jp


def plot_angles(img, name):

    fig = plt.figure(figsize=(40, 40))
    fig.suptitle(name, fontsize=16)

    ax1 = fig.add_subplot(331)
    ax1.title.set_text('edges')
    plt.imshow(img)

    ax2 = fig.add_subplot(332)
    ax2.title.set_text('rows')
    plt.plot(np.sum(img, axis=1))

    ax3 = fig.add_subplot(333)
    ax3.title.set_text('columns')
    plt.plot(np.sum(img, axis=0))

    for k in range(1, 7):
        angle = -k
        rotated = rotate(img, angle)
        x = np.sum(rotated, axis=1)
        ax = fig.add_subplot(3, 3, 3 + k)
        ax.title.set_text('rotated ' + str(angle) + " degrees")
        plt.plot(x)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def find_words(img, name, photo):
    # sumujemy białe piksele w rzędach i kolumnach
    rows = np.sum(img, axis=1)
    cols = np.sum(img, axis=0)

    # przycinamy outliery (niepotrzebne tło)
    gora, dol, lewo, prawo = przytnij(rows, cols)
    img = img[gora:dol, lewo:prawo]
    rows = rows[gora:dol]
    cols = cols[lewo:prawo]

    # usuwamy małe szumy
    rows_binary = rows > max(rows)/ 10
    rows_filtered = []
    for r in range(len(rows)):
        if rows_binary[r]:
            rows_filtered.append(rows[r])
        else:
            rows_filtered.append(0)

    rows_filtered = np.array(rows_filtered)

    # znajdujemy maksima lokalne
    maksima = peak_local_max(rows_filtered, 40)
    maksima = [x[0] for x in maksima]

    # odfiltrowujemy szumy
    maksima_filtered = []
    for i in range(len(maksima)-1):
        if abs(maksima[i] - maksima[i+1]) > 10:
            maksima_filtered.append(maksima[i])
    maksima_filtered.append(maksima[-1])

    img = img.astype(np.uint8)

    # wyznaczamy obszary dookoła maksimów - będą to nasze wiersze z tekstem
    bounds = get_rows(maksima_filtered, len(rows))

    # robimy dylatację w poziomie
    img = dilation(img, rectangle(1, 10))
    img2 = img.astype(np.uint8) * 255
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # definicje kolorków ;)
    colors = [(204, 255, 255), (0, 0, 204), (0, 204, 0), (255, 0, 0), (153, 0, 153), (255, 128, 0),
              (255, 204, 229), (255, 255, 51), (0, 128, 255), (153, 255, 153), (255, 102, 255), (255, 0, 127)]

    for ix, row in enumerate(bounds):
        color = colors[ix%len(colors)]

        # wycinamy z obrazka wiersz
        cut = img[row[0]:row[1]]
        cut = cut.astype(np.uint8)

        # znajdujemy na nim kontury
        to_unpack= cv2.findContours(cut, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(to_unpack)==3:
            _, contours, _ = to_unpack
        else:
            contours, _ = to_unpack

        contours = np.array(contours)

        # każdy kontur przybliżamy do prostokąta
        cont_edges = []
        for c in contours:

            cont = np.array(c[:, 0])
            left = min(cont[:,0]) + lewo # dodajemy lewo i górę, żeby zmapować na odpowiednie współrzędne w oryginalnym zdjęciu
            right = max(cont[:,0]) + lewo
            up = min(cont[:, 1]) + row[0] + gora
            down = max(cont[:, 1]) + row[0] +gora
            print(left, right, up, down)
            cont_edges.append([left, right, up, down])

            # rectangle za nc nie chciało mi zadziałać, więc póki co są 4 linie xD
            cv2.line(photo, (left, up), (right, up), color, 3)
            cv2.line(photo, (left, up), (left, down), color, 3)
            cv2.line(photo, (left, down), (right, down), color, 3)
            cv2.line(photo, (right, up), (right, down), color, 3)

        # ustawiamy w kolejności od lewej do prawej
        cont_edges = cont_edges.sort(key=lambda x: x[3])

    fig = plt.figure(figsize=(40, 40))
    fig.suptitle(name, fontsize=16)
    io.imshow(photo)
    plt.show()


def process(img_name):
    src = cv2.imread(img_name, cv2.IMREAD_COLOR)

    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    # robimy splot z maską, która odfiltruje poziome i pionowe linie (kratkę z kartki)
    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    gray2 = cv2.filter2D(gray, -1, kernel)
    gray2 = invert(gray2)
    gray2 = binary_dilation(gray2, disk(3))
    # img_edges = cv2.Canny(gray, 200, 200, apertureSize=3)

    plot_angles(gray2, img_name)
    find_words(gray2, img_name, src)


def sortKeyFunc(s):
    return int(os.path.basename(s)[4:-4])


if __name__ == "__main__":

    filenames = glob.glob(os.path.join(str(sys.argv[1]), "*.jpg"))

    if not filenames:
        raise ValueError('Błąd przy wczytywaniu plików')
    filenames.sort(key=sortKeyFunc)

    print(filenames)

    for i in reversed(range(len(filenames))):
        process(filenames[i])
