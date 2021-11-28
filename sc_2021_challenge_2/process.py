# import libraries here
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from fuzzywuzzy import fuzz

import neuralNetwork as nn
import imageManagment as im


def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran

    letters1 = im.return_trained_letters(train_image_paths[0])
    letters2 = im.return_trained_letters(train_image_paths[1])
    letters = []
    # for i in range(0, len(letters1)):  
    #     letters.append(i)

    # for i in range(0, len(letters2)):  
    #     letters.append(i)

    letters = letters1 + letters2

    #print(letters)
    print(len(letters1))
    print(len(letters2))

    x_train = im.prepare_for_ann(letters)
    y_train = im.convert_output(nn.make_alphabet())
    ann = nn.load_trained_ann()


    #print(x_train)
    #print(y_train)

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = nn.create_ann()
        ann = nn.train_ann(ann, x_train, y_train)
        print("Treniranje modela zavrseno.")
        nn.serialize_ann(ann)

    #model = None
    return ann


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    letters, k_means = im.return_letters_with_kmeans(image_path)
    prepared = im.prepare_for_ann(letters)
    predicted = trained_model.predict(np.array(prepared, np.float32))

    if k_means is None:
        extracted_text="I have error"
    else:
        extracted_text = im.display_result(predicted, nn.make_alphabet(), k_means)

    #print(extracted_text)

    fuzzy_extracted_text = ""
    for extracted_word in list(extracted_text.split(' ')):
        vocabulary_words = list(vocabulary.keys())
        same_word = False
        final_word = ""
        nearest_words = []
        lowest_distance=500
        for word in vocabulary_words:
            if word == extracted_word:
                final_word = word
                same_word = True
            else:
                distance = fuzz.ratio(word,extracted_word)
                if distance < lowest_distance:
                    lowest_distance = distance
                elif distance == lowest_distance:
                    nearest_words.append([word, vocabulary[word]])

        if not same_word:
            word_repetition = 0
            for word_distance_repet in nearest_words:
                if int(word_distance_repet[1])>word_repetition:
                    final_word=word_distance_repet[0]
                    word_repetition=int(word_distance_repet[1])


        if final_word is not None:
            fuzzy_extracted_text += final_word + " "
        else:
            fuzzy_extracted_text += extracted_word + ""
        fuzzy_extracted_text.rstrip() #za brisanje " " na kraju stringa
    
    print(fuzzy_extracted_text)

    return fuzzy_extracted_text
