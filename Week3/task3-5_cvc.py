""" 
CODI EXEMPLE:
Una vegada tinguis un mode entrenat per la tasca3 de patches, tindras un model que treurà:
         
        Una llista de arrays on cada array es la sortida d'una capa.
        El tema aquí esta en que ara el input no es una imatge sencera sino un patch.
"""
# carregar model
# crear els patches de una imatge. tant les de train com les de test
patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=num_patches)
out = model.predict(patches/255.)
"""
si vols treure l'accuracy del model hauras de fer un avg de la sortida dels patches
i despres veure quin es el més activat """
predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )

"""
Si en canvi vols construir uns descriptors crec que la solucio es construir la mateixa estructura:
(1881 imatges, X patches, Y features) 
per tant fer un for de les imatges
cridar la funcio que et fa els patches
i després per cada patch treure els features d'aquell patch
"""
out = model.predict(patches/255.)
"""
treura una llista de arrays on cada un sera els features d'un patch
hem de fer que el model nomes torni la capa que volem no totes com a la tasca 2
[feat_patch1, feat_patch2, ...]
llavors fer un array que sigui (1881, X patches, X features)
per cada imatge tindras X patches, i de cada patch el seu feature corresponent
"""
