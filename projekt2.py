import glob, os, sys
import cv2
import numpy as np
from skimage import io
from skimage.util import invert
from skimage.transform import rotate
from skimage.morphology import dilation, binary_dilation, disk, rectangle, binary_opening, binary_closing
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt


def get_rows(maxima, size):
    lines = sorted(maxima)
    halfs = [(lines[i] + lines[i+1])/2 for i in range(len(lines)-1)]
    bounds = []
    for j in range(len(lines)):
        if j == 0:
            bounds.append([0, int(halfs[0])])
        elif j == len(lines) - 1:
            lens = [b[1]-b[0] for b in bounds[1:]]
            bounds.append([int(halfs[-1]), int(halfs[-1]+ np.mean(lens))])
        else:
            bounds.append([int(halfs[j-1]), int(halfs[j])])

    bounds[0][0] = max(int(halfs[0] - np.mean(lens)), 0)

    row_lengths = [b[1]-b[0] for b in bounds]
    avg_len = int(np.mean(row_lengths))
    for_join = []

    for r, row in enumerate(row_lengths):
        if row > avg_len*1.4:
            bounds[r][1] = bounds[r][0] + avg_len
        if r < len(row_lengths) - 1 and row < avg_len * 0.6 and row_lengths[r + 1] < avg_len * 0.6:
            for_join.append(r+1)
            
    for k in for_join:
        bounds[k-1][1] = bounds[k][1]
    # trzeba zrobić kolejną iterację, bo operujemy na indeksach,
    # więc nie możemy usuwać rzeczy z listy w trakcie przetwarzania
    for k in reversed(for_join):
        bounds.pop(k)

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

def find_words(img, name, photo, ifprint = False):

    original = photo.copy()

    # sumujemy białe piksele w rzędach i kolumnach
    rows = np.sum(img, axis=1)
    cols = np.sum(img, axis=0)

    # przycinamy outliery (niepotrzebne tło)
    gora, dol, lewo, prawo = przytnij(rows, cols)
    img = img[gora:dol, lewo:prawo]
    rows = rows[gora:dol]
    cols = cols[lewo:prawo]

    masked_image = np.zeros_like(img, dtype=np.float32)

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

    for ix, row in enumerate(bounds):
        mask = np.zeros_like(img)

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
            down = max(cont[:, 1]) + row[0] + gora

            cont_edges.append([left, right, up, down])

            if (right-left)*(down-up) >300:
                cv2.rectangle(mask,(left, up), (right, down), 1, -1)
        
        #TODO tu mi się coś nie zgadza, tw dwie wielkości, moze trochę nie rozumiem tamtej manipulacji wielkościami?
        mask = binary_closing(mask, rectangle(5,10))
        # rozciągamy zaznaczenie w pionie
        mask_cols = np.sum(mask, axis=0)
        row_beg = row[0] + gora
        row_end = row[1] + gora
        for nr_col, col in enumerate(mask_cols):
            if col > 0:
                mask[row_beg:row_end, nr_col] = 1

        # jesli zaznaczenie w wierszu ma powierzchnię < 2%, to zerujemy wiersz
        pixel_sum = np.sum(mask[row_beg:row_end])
        if pixel_sum < (row_end - row_beg)*len(cols) * 0.02:
            mask[row_beg:row_end] = 0

        #masked_image[mask] = ix % 2 * 0.5 + 0.5
        masked_image[mask] = ix + 1

        # ustawiamy w kolejności od lewej do prawej
        # cont_edges = cont_edges.sort(key=lambda x: x[3])

    masked_image = cv2.copyMakeBorder(
        masked_image,
        top=0,
        bottom=photo.shape[0]-masked_image.shape[0],
        left=0,
        right=photo.shape[1]-masked_image.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    if ifprint:
        fig = plt.figure(figsize=(40, 40))
        fig.add_subplot(1, 2, 1)
        io.imshow(masked_image)
        fig.add_subplot(1, 2, 2)
        io.imshow(original)
        fig.suptitle(name, fontsize=16)
        plt.show()
    return masked_image


def process(img_name, ifprint):
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

    #if ifprint:
    #    plot_angles(gray2, img_name)

    return find_words(gray2, img_name, src, ifprint)


def sortKeyFunc(s):
    return int(os.path.basename(s)[4:-4])


def detect_words(path_to_image, ifprint = False):
    return process(path_to_image, ifprint)


if __name__ == "__main__":

    filenames = glob.glob(os.path.join(str(sys.argv[1]), "*.jpg"))

    if not filenames:
        raise ValueError('Błąd przy wczytywaniu plików')
    filenames.sort(key=sortKeyFunc)

    #print(filenames)

    for i in range(len(filenames)):
        #use ifprint = True to check result  
        detect_words(filenames[i])
