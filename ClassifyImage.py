import pandas as pd
import numpy as np
import PIL 
from PIL import Image
import base64
import os
import re
from StringIO import StringIO
import pylab as pl 
from yhat import YhatModel,Yhat,preprocess
from sklearn.decomposition import RandomizedPCA,PCA
from sklearn.neighbors import KNeighborsClassifier



#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (300, 167)
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = PIL.Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


    # TODO PATH TO YOUR DATA
img_dir = "/home/mohamed/Downloads/images/"
images = [img_dir+ f for f in os.listdir(img_dir)]
notSigned="not"+" "+"signed-document"
labels = [notSigned if notSigned in f.split('/')[-1] else "signed-document" for f in images]

data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

data = np.array(data)
print data


is_train = np.random.uniform(0, 1, len(data)) <= 0.7
y = np.where(np.array(labels)==notSigned, 1, 0)

train_x, train_y = data[is_train], y[is_train]
test_x, test_y = data[is_train==False], y[is_train==False]





pca = PCA(svd_solver='randomized', n_components=2)
X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "not signed-document", "signed-document")})
colors = ["red", "yellow"]
for label, color in zip(df['label'].unique(), colors):
    mask = df['label']==label
    pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
pl.legend()
pl.show()


pca = PCA(svd_solver='randomized',n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

print train_x[:5]


knn = KNeighborsClassifier()
knn.fit(train_x, train_y)



def string_to_img(image_string):
    print "called string_to_image"
    #we need to decode the image from base64
    image_string = base64.decodestring(image_string)
    #since we're seing this as a JSON string, we use StringIO so it acts like a file
    img = StringIO(image_string)
    img = PIL.Image.open(img)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return pca.transform(img_wide[0])

def classify_image(data):
    print "called classify_image"
    preds = knn.predict(data)
    preds = np.where(preds==1, "not signed-document", "signed-document")
    pred = preds[0]
    return {"image_label": pred}




class ImageClassifier(YhatModel):
    REQUIREMENTS = [
      "PIL==1.1.7"
    ]
    
    def execute(self, data):
        print "called execute"
        img_string = data.get("image_as_base64_string", None)
        if img_string is None:
            return {"status": "error", "message": "data was None", "input_data": data}
        else:
            img = string_to_img(img_string)
            pred = classify_image(img)
            return pred