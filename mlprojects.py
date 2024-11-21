# These are the libraries of python which I will be using here
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Here using cv2 library I will be scanning the image and get its dimensions and resizing its dimensions
im_path="../camscanner/bill.jpg"
img=cv2.imread(im_path)
print(img.shape)
img=cv2.resize(img,(1000,800))
print(img.shape)
plt.imshow(img)
plt.show()

# Here we will make our document becoming grayed out and then blurred because we have to obtain the actual part to be getting

orig=img.copy()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='binary')
plt.show()
blurred=cv2.GaussianBlur(gray,(5,5),0)
plt.imshow(blurred,cmap='binary')
plt.show()

# Here we will print the results of the blurred documents 

regen=cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
plt.imshow(orig)
plt.show()
plt.imshow(regen)
plt.show()

# Here we will detedt the edges of the contents present in the image 

edge=cv2.Canny(blurred,0,50)
orig_edge=edge.copy()
plt.imshow(orig_edge)
plt.title("Edge Detection")
plt.show()

edge.shape

# Approximate the countors of the documnents of the image


contours,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print(len(contours))
contours=sorted(contours,reverse=True,key=cv2.contourArea)

for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)
    if len(approx)==4:
        d=approx
        break
print(d.shape)

# Here I am reordering the points for the image detedtion
def reorder(h):
  h=h.reshape((4,2))
  print(h)
  hnew=np.zeros((4,2),dtype=np.float32)
  add=h.sum(axis=1)
  hnew[3]=h[np.argmin(add)]
  hnew[1]=h[np.argmax(add)]
  diff=np.diff(h,axis=1)
  hnew[0]=h[np.argmin(diff)]
  hnew[2]=h[np.argmax(diff)]
  return hnew

reorder = reorder(d)
print("*********************")
print(reorder)

# Here I am transforming the perspective of the image

input_represent=reorder
output_map=np.float32([[0,0],[800,0],[800,800],[0,800]])

M=cv2.getPerspectiveTransform(input_represent,output_map)
ans=cv2.warpPerspective(orig,M,(800,800))

plt.imshow(ans)
plt.show()

# Convert the transformed image into grayscale and blurred format

res= cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
b_res=cv2.GaussianBlur(res,(3,3),0)
plt.show()
plt.imshow(res,cmap='binary')
plt.show()