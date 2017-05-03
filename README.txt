It's advised for you to use QT Creator and OpenCV libs (Vlfeat, slic code found on the internet, etc were also used).

This code aims into creating an arff file that extract characteristics from an image using the Game of Life theory, the image is split into superpixels which will be later on classified by you according to the classes you want. Having the arff file, you will be able to train a classificator (use Weka) and have your own Game of Life Superpixel classificator!

First you need to set the build directory as the same directory that has the "Arquivos .arff" "Images" and "weks-results" folder and check your pro files to connect all the libraries needed.

After that you will be able to run the program after setting all the paramaters that you wish in the code.

The main function is "generateArff" which is called using "/Arquivos .arff/" and '3' paramaters, keep it that way if you with to use the arff generator for superpixel classification with cellular automata.

When running the program, you will need to manually classify each superpixel from each image that you set at the "for" loop by just typing the number of the class and pressing enter (this takes time, so you shouldn't use lots of images).

The arff will be generated (values are normalized), you can use the arff file in Weka to train a classificator.

It's also generated a file that has only extracted color values from the superpixel for compairson.

The code is commented but the main paramaters you should be aware of: 
	Image location folder
	Arff file name
	Regularization value (keep high, it works poorly at low values)
	Region size
	Arff generated classes: search for "@attribute class" in the code, what comes next are the classes that you will be using (use numbers)
	Index range of the images: since the images are saved in the image folder as numbers, you should find the "for" loop into the "generateArff" function which has the index paramater

The paramater 1,2,4 from the generateArff function are not from great use, so they are not explained here.

It is required that you have a knowledge of image processing and other concepts that are used in this work.

For better understanding there is a conclusion work course which has all the theory, methods, credits, etc, etc.

If you wish to contact me or want to understand more about this work, send me an e-mail: gustavor@alunos.utfpr.edu.br

THE CODE IS NOT OPTIMIZED NOR HAVE A USER-FRIENDLY INTERFACE


	