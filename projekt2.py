import glob, os, sys
import cv2
import numpy as np
from skimage import io
from skimage.util import invert
from skimage.transform import rotate
from skimage.morphology import dilation, binary_dilation, disk, rectangle, binary_opening, binary_closing
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from box_line_removal import line_removal
import keras
import math


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

def pad_scale(image):
    if image.shape[0] > image.shape[1]:
        scale_factor = 28 / image.shape[0]
        new_x = round(image.shape[0] * scale_factor)
        image = cv2.resize(image, dsize=(28, new_x), fx=scale_factor, fy=scale_factor)
        image = cv2.copyMakeBorder(image, top=0, bottom=0, left=math.floor((28-new_x)/2), right=math.ceil((28-new_x)/2), borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        scale_factor = 28 / image.shape[1]
        new_y = round(image.shape[1] * scale_factor)
        image = cv2.resize(image, dsize=(new_y, 28), fx=scale_factor, fy=scale_factor)
        image = cv2.copyMakeBorder(image, top=math.floor((28-new_y)/2), bottom=math.ceil((28-new_y)/2), left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)
    return image


def find_words(img, name, photo, ifprint = False):

    #TODO
    
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
        
        # TODO tu mi się coś nie zgadza, tw dwie wielkości, moze trochę nie rozumiem tamtej manipulacji wielkościami?
        mask = binary_closing(mask, rectangle(5, 10))
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

    rows_bounds = [[r[0]+gora, r[1]+gora] for r in bounds]

    return photo, masked_image, rows_bounds

def get_index(img, masked_image, bounds, model):
    result = []
    for i, r in enumerate(bounds):
        row_result = []

        row = img[r[0]:r[1]]
        height = r[1] - r[0]
        masked_row = masked_image[r[0]:r[1]]
        sum_col = np.sum(masked_row, axis=0)

        edges = [ix for ix in range(len(sum_col)-1) if sum_col[ix+1] != sum_col[ix]]
        if edges:
            width = edges[-1] - edges[-2]
            index_area = row[:, edges[-2]:edges[-1]]

            try:
                th = line_removal(index_area)
            except:
                #print('no lines found')
                continue

            cols_hist = np.sum(th, axis=0)/255

            for c in range(len(cols_hist)):
                if cols_hist[c] >= height * 0.6:
                    surround = cols_hist[max(0,c-6): min(c+6, width)]
                    surround = [s for s in surround if s < height * 0.6]
                    cols_hist[c] = int(np.mean(surround))

            from scipy.ndimage.filters import gaussian_filter1d

            hist_smooth = gaussian_filter1d(cols_hist, sigma=4)

            maksima = peak_local_max(hist_smooth, 5)
            maksima = sorted([x[0] for x in maksima])

            gaps = [maksima[o+1]-maksima[o] for o in range(len(maksima)-1)]
            max_filtered = []
            flag=0
            for ig, g in enumerate(gaps):
                if flag == 1:
                    flag = 0
                    continue
                if g > 0.7*np.mean(sorted(gaps)[1:]):
                    max_filtered.append(maksima[ig])
                else:
                    max_filtered.append(int((maksima[ig]+maksima[ig+1])/2))
                    flag = 1
            if len(maksima) > 0:
                max_filtered.append(maksima[-1])

            #plt.plot(hist_smooth)

            digits_bounds = []

            for im, m in enumerate(max_filtered):
                if im == 0:
                    digits_bounds.append(0)
                else:
                    digits_bounds.append(int((max_filtered[im-1] + max_filtered[im])/2))
                    if im == len(max_filtered) - 1:
                        digits_bounds.append(width)

            kernel = np.zeros((2,2) ,np.uint8)
            kernel2 = np.zeros((2,2),np.uint8)
            for i in (0,1):
                kernel[1][i] = 1
                kernel2[i][1] = 1

            for k in range(len(max_filtered)-1):
                fragment = th[:, digits_bounds[k]:digits_bounds[k+1]]
                fragment = cv2.morphologyEx(fragment, cv2.MORPH_OPEN, kernel)
                fragment = cv2.morphologyEx(fragment, cv2.MORPH_OPEN, kernel2)

                countver = np.sum(fragment, axis = 1)
                counthor = np.sum(fragment, axis = 0)
                maxcounthor = (fragment.shape[1] * 255) * 0.8
                for i, count in enumerate(counthor):
                    if i < fragment.shape[1]*0.2 or i > fragment.shape[1]*0.8:
                        if count >= maxcounthor:
                            for r in range(fragment.shape[0]):
                                fragment[r][i] = 0
                to_cut_top = 0
                to_cut_bottom = fragment.shape[0]-1
                for i, count in enumerate(countver):
                    if i < fragment.shape[0]*0.2:
                        if count == 0: to_cut_top=i
                    if i > fragment.shape[0]*0.5:
                        if count < 500: 
                            to_cut_bottom=i
                            break

            
                fragment = fragment[to_cut_top : to_cut_bottom, :]
                fragment = pad_scale(fragment)

                to_predict = [fragment]
                to_predict = np.asarray(to_predict).astype('float32')
                to_predict = np.expand_dims(to_predict, axis=3)
                to_predict /= 255
                classes = model.predict_classes(to_predict)

                row_result.append(str(classes[0]))
        indeks = ''.join(row_result)
        result.append((None, None, indeks))

    return result


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

    return find_words(gray2, img_name, src, ifprint)


def sortKeyFunc(s):
    return int(os.path.basename(s)[4:-4])


def detect_words(path_to_image, ifprint = False):
    _, masked_image, _ = process(path_to_image, ifprint)
    return masked_image


def ocr(path_to_image):
    img, masked_image, rows_bounds = process(path_to_image, ifprint=False)
    model = keras.models.load_model('kares_model.mod')
    result = get_index(img, masked_image, rows_bounds, model)
    print(result)
    return result


if __name__ == "__main__":
    filenames = glob.glob(os.path.join(str(sys.argv[1]), "*.jpg"))

    if not filenames:
        raise ValueError('Błąd przy wczytywaniu plików')
    filenames.sort(key=sortKeyFunc)

    for i in range(len(filenames)):
        ocr(filenames[i])
