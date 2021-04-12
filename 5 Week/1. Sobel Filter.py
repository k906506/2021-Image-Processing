import numpy as np

def get_sobel():
    derivative = np.array([[-1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivative)
    y = np.dot(derivative.T, blur.T)

    return  x, y

def main():
    sobel_x, sobel_y = get_sobel()
    print('sobel_x')
    print(sobel_x)

    print('sobel_y')
    print(sobel_y)

if __name__ == "__main__":
    main()