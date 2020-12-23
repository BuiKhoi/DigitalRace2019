import os
import cv2

#get images
image_folder = './image_data/2020-12-10 10:41/'
images = []
for sub_folder in os.listdir(image_folder):
    sub_folder = image_folder + sub_folder + '/'
    for file in os.listdir(sub_folder):
        images.append(sub_folder + file)
        
#sort images
images.sort(key=lambda x: int(x.split('/')[-1].split('x')[0]))

#show images
for image in images:
    _, spd, ste = image.split('x')
    ste = ste.split('.')[0]
    image = cv2.imread(image)
    iamge = cv2.putText(image, ste, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('image', image)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()