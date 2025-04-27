
import numpy as np
import os
from PIL import Image


NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)



def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


    


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions 
        if not isinstance(pixels, list) or not all([isinstance(row, list) \
            for row in pixels]):
            raise TypeError()

        if len(pixels) == 0 or any(len(row) != len(pixels[0]) \
            for row in pixels):
            raise TypeError()
        row_length = len(pixels[0])
        for row in pixels:
            if len(row) != row_length:
                    raise TypeError()
            for pixel in row:
                if not isinstance(pixel, list) or len(pixel) != 3:
                    raise TypeError()
                if not all(isinstance(value, int) and 0 <= value <= 255 for value in pixel):
                    raise ValueError()

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])



    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        # YOUR CODE GOES HERE #
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        # YOUR CODE GOES HERE #
        deep_copy_pixels = [[[k for k in j]for j in i] for i in self.pixels ]
        return deep_copy_pixels

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        deep_copy_image = RGBImage(self.get_pixels())
        return deep_copy_image

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # YOUR CODE GOES HERE #
        if type(row) != int or type(col) != int:
            raise TypeError()
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            raise ValueError
        return tuple(self.pixels[row][col])


    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #

        if not isinstance(new_color, tuple) or len(new_color) != 3 or \
           not all(isinstance(value, int) and -2 < value <= 255 for value in new_color):
            raise ValueError

        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            raise ValueError

        if new_color[0] != -1:
            self.pixels[row][col][0] = new_color[0]
        if new_color[1] != -1:
            self.pixels[row][col][1] = new_color[1]
        if new_color[2] != -1:
            self.pixels[row][col][2] = new_color[2]







# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #
        image.pixels = [[[255-k for k in j]for j in i] for i in image.pixels]
        
        return RGBImage(image.pixels)


    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        image.pixels = [[[sum(j)//3 for k in j]for j in i] \
        for i in image.pixels]

        return image 

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        image.pixels = [row[::-1] for row in image.pixels[::-1]]
        return image

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        # YOUR CODE GOES HERE #
        total = image.num_rows * image.num_cols
        brightness = sum(sum(pixel)//3 for row in image.pixels \
            for pixel in row)
        return brightness // total

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(intensity, int):
            raise TypeError()
        if not -255 <= intensity <= 255:
            raise ValueError()

        
        image.pixels = [
            [
                [
                    max(0, min(255, channel + intensity))
                    for channel in pixel
                ]
                for pixel in row
            ]
            for row in image.pixels
        ]


        return image


    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        # YOUR CODE GOES HERE #
        blurred_pixels = []
        for row in range(image.num_rows):
            blurred_row = []
            for col in range(image.num_cols):
                p_neighbors = [
                    [row, col], [row-1, col-1], [row-1, col], [row-1, col+1], 
                    [row+1, col-1], [row+1, col], [row+1, col+1], 
                    [row, col-1], [row, col+1]
                ]
                p_neighbors = [i for i in p_neighbors if 0 <= i[0] < \
                image.num_rows and 0 <= i[1] < image.num_cols]
                neighbors_pixels = [image.pixels[r][c] for r, c in p_neighbors]
                red = sum([i[0] for i in neighbors_pixels]) // \
                len(neighbors_pixels)
                green = sum([i[1] for i in neighbors_pixels]) // \
                len(neighbors_pixels)
                blue = sum([i[2] for i in neighbors_pixels]) // \
                len(neighbors_pixels)
                blurred_row.append([red, green, blue])
            blurred_pixels.append(blurred_row)
        
        return RGBImage(blurred_pixels)

# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupon = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
    
        if self.coupon == 0:
            self.cost += 5
        else:
            self.coupon -= 1
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #
        
        if self.coupon == 0:
            self.cost += 6
        else:
            self.coupon -= 1
        return super().negate(image)
    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        super().rotate_180(image)
        if self.coupon == 0:
            self.cost += 10
        else:
            self.coupon -= 1

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #

        
        if self.coupon == 0:
            self.cost += 1
        else:
            self.coupon -= 1
        return super().adjust_brightness(image)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        
        if self.coupon == 0:
            self.cost += 5
        else:
            self.coupon -= 1

        return super().blur(image)
    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        elif amount <= 0:
            raise ValueError()

        self.coupon = amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def tile(self, image, new_width, new_height):
        """
        Returns a new image with size new_width x new_height where the
        given image is tiled to fill the new space

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> new_width, new_height = 70, 70
        >>> img_exp = img_read_helper('img/exp/square_32x32_tile.png')
        >>> img_tile = img_proc.tile(img_in, new_width, new_height)
        >>> img_tile.pixels == img_exp.pixels # Check tile output
        True
        >>> img_save_helper('img/out/square_32x32_tile.png', img_tile)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(image,RGBImage):
            raise TypeError()
        elif not isinstance(new_width, int) or not isinstance(new_height,int):
            raise TypeError()
        elif new_width < image.num_cols or new_height < image.num_rows:
            raise ValueError()

        new_image = [[[0,0,0] for w in range(new_width)]for h \
        in range(new_height)]

        for row in range(new_height):
            for col in range(new_width):
                ori_row = row%image.num_rows
                ori_col = col%image.num_cols
                new_image[row][col] = image.pixels[ori_row][ori_col]

        return RGBImage(new_image)


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(sticker_image, RGBImage) or not \
        isinstance(background_image, RGBImage):
            raise TypeError()
        elif sticker_image.num_cols > background_image.num_cols or\
        sticker_image.num_rows > background_image.num_rows:
            raise ValueError()
        elif not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        elif x_pos < 0 or y_pos <0:
            raise ValueError()
        elif background_image.num_cols - x_pos < sticker_image.num_cols or \
        background_image.num_rows - y_pos < sticker_image.num_rows:
            raise ValueError()

        new_pixels = [[[0,0,0] for w in range(background_image.num_rows)]\
        for h in range(background_image.num_cols)]

        for r in range(sticker_image.num_rows):
            for c in range(sticker_image.num_cols):
                background_image.pixels[r+y_pos][c+x_pos] = \
                sticker_image.pixels[r][c]

        return RGBImage(background_image.pixels)




    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #

        kernel = [-1,-1,-1,-1,-1,-1,-1,-1]


        averaged_pixels = [[[0] for r in range(image.num_rows)] \
        for c in range(image.num_cols)]
        masked_pixels = [[[0,0,0] for r in range(image.num_rows)] \
        for c in range(image.num_cols)]

        averaged_pixels = [
            [sum(image.pixels[row][col]) // 3 for col in range(image.num_cols)]
            for row in range(image.num_rows)
        ]

        for row in range(image.num_rows):
            for col in range(image.num_cols):
                p_all = [
                    [row+1, col-1],[row+1, col],[row+1, col+1],\
                    [row, col-1],[row, col+1],\
                    [row-1, col-1],[row-1, col], [row-1, col+1]
                ]

                p_valid = [i for i in p_all if 0 <= i[0] and i[0] < \
                image.num_rows and 0 <= i[1] and i[1] < image.num_cols]

                p_values = [averaged_pixels[i[0]][i[1]] for i in p_valid]

                masked_p_values = sum(p_values)


                masked_value = averaged_pixels[row][col]*8 - masked_p_values

          
                masked_value = max(0, min(255, masked_value))
         
                masked_pixels[row][col] = [masked_value,masked_value,masked_value]
        return RGBImage(masked_pixels)



# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        # YOUR CODE GOES HERE #
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #

        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        elif image1.num_cols != image2.num_cols or image1.num_rows != image2.num_rows:
            raise ValueError()

        distance_total = sum([sum((c1 - c2) ** 2 for c1, c2 in zip(image1.pixels[row][col], image2.pixels[row][col]))\
        for row in range(image1.num_rows) for col in range(image1.num_cols)])**0.5

        return distance_total


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        # YOUR CODE GOES HERE #

        list_count = [candidates.count(i) for i in candidates]
        max_index = list_count.index(max(list_count))
        return candidates[max_index]


    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #


        if self.data is None:
            raise ValueError()
        distances = [self.distance(data_image,image) for data_image, label in self.data]
        valid_indexes = sorted(range(len(distances)), key = lambda i: distances[i])[:self.k_neighbors]
        nearest_labels = [self.data[i][1] for i in valid_indexes]
        predicted_label = self.vote(nearest_labels)
        return predicted_label



def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label

