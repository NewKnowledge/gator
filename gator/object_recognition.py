from nk_imagenet import ImagenetRecognizer

recognizer = None
def init(model='inception_v3', n_objects=10):
    global recognizer
    recognizer = ImagenetRecognizer(model=model, n_objects=n_objects)
    

def predict(images_array):
    ''' takes a batch of images as a 4-d array and returns the detected objects as a list of {object: score} dicts '''
    if not recognizer:
        init()

    return recognizer.get_objects(images_array)
    
