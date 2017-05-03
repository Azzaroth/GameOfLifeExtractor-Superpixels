#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <unistd.h>
#include <fstream>
#include <string>
#include <math.h>

extern "C" {
    #include "slic.h"
    #include "generic.h"
}

using namespace cv;
using namespace std;

void distanceTransform(Mat src);
void normalizedHistogram(int** matrix, int** other_matrix, int** color_matrix, int regions, float** normalized_matrix, float** other_normalized_matrix, float** color_normalized_matrix);

//Return the directory of the project
String getUserDir() {
    char user_dir[1024];
    if (getcwd(user_dir, sizeof(user_dir)) != NULL)
        return user_dir;
    else
        perror("getcwd() error");
    return "Not found";
}

//Return the class of the image given an index (no superpixel)
String imageClass(int index) {
    if (index >= 0 && index < 100)
        return "indio";
    else if (index >= 100 && index < 200)
        return "praia";
    else if (index >= 200 && index < 300)
        return "monumento";
    else if (index >= 300 && index < 400)
        return "onibus";
    else if (index >= 400 && index < 500)
        return "dinossauro";
    else if (index >= 500 && index < 600)
        return "elefante";
    else if (index >= 600 && index < 700)
        return "flor";
    else if (index >= 700 && index < 800)
        return "cavalo";
    else if (index >= 800 && index < 900)
        return "montanha";
    return "comida";
}

//Used for non-superpixel algorithm
int neighborhoodEquals(int* neighborhood) {

    int count = 0;

    for (int i = 0; i < 9; i++) {
        if (i != 4 && (neighborhood[i] == neighborhood[4])) count++;
    }

    return count;
}

//Used for non-superpixel algorithm
array<int,255> stillAliveNoSP(Mat image) {

    int neighborhood[3][3];
    int count, up, down, left, right;
    array<int, 255> stillAliveVector;

    for (int z = 0; z < 255; z++) {
        stillAliveVector[z] = 0;
    }

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {

            count = 0;

            up = i - 1;
            down = i + 1;
            left = j - 1;
            right = j + 1;

            if (i == 0) up = image.rows - 1;
            else if (i == (image.rows - 1)) down = 0;
            if (j == 0) left = image.cols - 1;
            else if (j == (image.cols - 1)) right = 0;

            neighborhood[0][0] = image.at<uchar>(up,left);
            neighborhood[0][1] = image.at<uchar>(up,j);
            neighborhood[0][2] = image.at<uchar>(up,right);
            neighborhood[1][0] = image.at<uchar>(i,left);
            neighborhood[1][1] = image.at<uchar>(i,j);
            neighborhood[1][2] = image.at<uchar>(i,right);
            neighborhood[2][0] = image.at<uchar>(down,left);
            neighborhood[2][1] = image.at<uchar>(down,j);
            neighborhood[2][2] = image.at<uchar>(down,right);

            count = neighborhoodEquals((int*)neighborhood);

            if (count == 2 || count == 3) {
                stillAliveVector[neighborhood[1][1]] += 1;
            }
        }
    }

    return stillAliveVector;

}

int generateSlicColor(Mat mat, vl_uint32* segmentation, bool borders) {

    // Convert image to one-dimensional array.
       float* image = new float[mat.rows*mat.cols*mat.channels()];
       for (int i = 0; i < mat.rows; ++i) {
           for (int j = 0; j < mat.cols; ++j) {
               // Assuming three channels ...
               image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
               image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
               image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
           }
       }

       // The algorithm will store the final segmentation in an one-dimensional array.
       vl_size height = mat.rows;
       vl_size width = mat.cols;
       vl_size channels = mat.channels();

       // The region size defines the number of superpixels obtained.
       // Regularization describes a trade-off between the color term and the
       // spatial term.
       // Good regularization value found was 10000 (You should use a high value)
       vl_size region = 80;
       float regularization = 10000;
       vl_size minRegion = 60;

       vl_size const numRegionsX = (vl_size) ceil((double) width / region) ;
       vl_size const numRegionsY = (vl_size) ceil((double) height / region) ;
       vl_size const numRegions = numRegionsX * numRegionsY ;

       //Function that segment the image using SLIC
       vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);


       //If set to true, will show borders in the image
       if (borders == true){

           // Convert segmentation.
           int** labels = new int*[mat.rows];
           for (int i = 0; i < mat.rows; ++i) {
               labels[i] = new int[mat.cols];

               for (int j = 0; j < mat.cols; ++j) {
                   labels[i][j] = (int) segmentation[j + mat.cols*i];
               }
           }

           int label = 0;
           int labelTop = -1;
           int labelBottom = -1;
           int labelLeft = -1;
           int labelRight = -1;

           for (int i = 0; i < mat.rows; i++) {
               for (int j = 0; j < mat.cols; j++) {

                   label = labels[i][j];

                   labelTop = label;
                   if (i > 0) {
                       labelTop = labels[i - 1][j];
                   }

                   labelBottom = label;
                   if (i < mat.rows - 1) {
                       labelBottom = labels[i + 1][j];
                   }

                   labelLeft = label;
                   if (j > 0) {
                       labelLeft = labels[i][j - 1];
                   }

                   labelRight = label;
                   if (j < mat.cols - 1) {
                       labelRight = labels[i][j + 1];
                   }

                   if (label != labelTop || label != labelBottom || label!= labelLeft || label != labelRight) {
                       mat.at<cv::Vec3b>(i, j)[0] = 0;
                       mat.at<cv::Vec3b>(i, j)[1] = 0;
                       mat.at<cv::Vec3b>(i, j)[2] = 255;
                   }
               }
           }
       }

       return numRegions;
}

String supervisionedString(int index) {

    switch(index){
    case 1:
        return "ceu";
        break;
    case 2:
        return "cidade";
        break;
    default:
        return "no index";
    }

    return "null";
}

//This will check if the given value is an actual number
inline bool isInteger(const string & s)
{
   if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;

   char * p ;
   strtol(s.c_str(), &p, 10) ;

   return (*p == 0) ;
}

//This function will be used to manually classify superpixels, so after training uniform results
//the tester will be able to know if the result is accurate in a normal picture
int* supervisionedClass (Mat mat, vl_uint32* segmentation, int regions) {

    int* vector = (int *)malloc(regions * sizeof(int *));
    Mat splitted[3];
    Mat merged;
    Mat returned;
    int rows = mat.rows;
    int cols = mat.cols;
    int index = 0;
    int value = -1;
    string input;

    cvtColor(mat,returned,CV_Lab2RGB);

    //Segmentation array will be assigned to this matrix
    int segment_matrix[rows][cols];

    //Assigning values to segment matrix from segment array
    for (int o = 0; o < rows; o++) {
        for (int l = 0; l < cols; l++) {
            segment_matrix[o][l] = segmentation[index];
            index++;
        }
    }

    for (int sp = 0; sp < regions; sp++) {
        input = 'a';
        split(mat,splitted);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (segment_matrix[i][j] != sp) {
                    splitted[0].at<uchar>(i,j) = 255;
                    splitted[1].at<uchar>(i,j) = 255;
                    splitted[2].at<uchar>(i,j) = 255;
                }
            }
        }
        merge(splitted,3,merged);
        imshow("Normal",returned);
        imshow("Super-pixel step classification", merged);
        waitKey(500);
        while (!isInteger(input)) {
        cin >> input;
        }
        value = stoi(input);
        vector[sp] = value;
    }

    return vector;

}

/*void test (Mat mat) {
int a;
Mat aux[3];
split(mat,aux);
for(int num = 0; num < 3; num++) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            a = aux[num].at<uchar>(i,j);
            cout << a << "\n";
        }
    }
}
waitKey(0);
}*/

void aliveRGBMatrix(Mat mat, int regions, vl_uint32* segmentation, int** matrix, int** other_matrix, int** color_matrix, float** normalized_matrix, float** other_normalized_matrix, float** color_normalized_matrix) {

    //mat will be split in three colors
    Mat rgb[3];

    int rows = mat.rows;
    int cols = mat.cols;

    //Segmentation array index iterator
    int index = 0;


    //Indexes to be used while assigning values to the characteristic matrix
    int superpixel;
    int value;

    //Segmentation array will be assigned to this matrix
    int segment_matrix[rows][cols];

    //Assigning values to segment matrix from segment array
    for (int o = 0; o < rows; o++) {
        for (int l = 0; l < cols; l++) {
            segment_matrix[o][l] = segmentation[index];
            index++;
        }
    }

    //Auxiliar variable to check if the pixel will live or die (Game of Life logic)
    int count = 0;

    //Clearing characteristic matrix
    for (int i = 0; i < regions*3; i++) {
        for (int j = 0; j < 32; j++) {
            matrix[i][j] = 0;
        }
    }

    //Clearing other characteristic matrix
    if (other_matrix != NULL) {
        for (int i = 0; i < regions*3; i++) {
            for (int j = 0; j < 32; j++) {
                other_matrix[i][j] = 0;
            }
        }
    }

    //Spliting original mat array into three chanells rgb[3] array
    split(mat,rgb);
    //Iterating throught each pixel
    //Checking their neighborhood
    //Comparing if their neighbor is not a border or in in its same Super-Pixel
    //If so, the pixel will be afected by Game of life's logic to see if it will live/die
    //If the pixel lives, the characteristic matrix will receive a PLUS ONE in the pixel intensity position on its
    //super pixel*color row
    for (int num = 0; num < 3; num++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                superpixel = segment_matrix[i][j] + (num*regions);
                value = rgb[num].at<uchar>(i,j)/8;
                color_matrix[superpixel][value]++;
                for (int k = i-1; k <= i+1; k++){
                    for (int z = j-1; z <= j+1; z++) {
                        if (k >= 0 && z >= 0 && k <= rows && z <= cols) {
                            if (((segment_matrix[k][z]/8) == (segment_matrix[i][j]/8)) && (k != i || z != j)) {
                                if (rgb[num].at<uchar>(i,j) == rgb[num].at<uchar>(k,z))
                                count++;
                            }
                        }
                    }
                }
                if (count == 2 || count == 3) {
                    //superpixel = segment_matrix[i][j] + (num*regions);
                    //value = rgb[num].at<uchar>(i,j)/8;
                    matrix[superpixel][value]++;
                }
                else {
                    //superpixel = segment_matrix[i][j] + (num*regions);
                    //value = rgb[num].at<uchar>(i,j)/8;
                    other_matrix[superpixel][value]++;
                }
                count = 0;
            }
        }
    }

    normalizedHistogram(matrix,other_matrix,color_matrix,regions,normalized_matrix,other_normalized_matrix,color_normalized_matrix);

}

int** allocateMatrix(int regions) {

    //Allocating characteristics matrix
    int** matrix = (int **)malloc(regions * 3 * sizeof(int *));

    for (int row = 0; row < regions*3; row++) {
        matrix[row] = (int *)malloc(32 * sizeof(int));
    }

    return matrix;
}

//Free space
void freeMatrix (int** matrix, int rows) {

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

//Free space
void freeFloatMatrix (float** matrix, int rows) {

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

float** allocateFloatMatrix(int regions) {

    //Allocating characteristics matrix
    float** matrix = (float **)malloc(regions * 3 * sizeof(float *));

    for (int row = 0; row < regions*3; row++) {
        matrix[row] = (float *)malloc(32 * sizeof(float));
    }

    return matrix;
}

void normalizedHistogram(int** matrix, int** other_matrix, int** color_matrix, int regions, float** normalized_matrix, float** other_normalized_matrix, float** color_normalized_matrix) {

    float total;
    float aux;

    for (int num = 0; num < 3; num++) {
        for (int i = 0; i < regions; i++) {
            for (int j = 0; j < 32; j++) {
                total = total + matrix[i + regions*num][j];
            }

            for (int j = 0; j < 32; j++) {
                aux = matrix[i + regions*num][j];
                normalized_matrix[i + regions*num][j] = aux/(total);
            }
            total = 0;
        }
    }

    for (int num = 0; num < 3; num++) {
        for (int i = 0; i < regions; i++) {
            for (int j = 0; j < 32; j++) {
                total = total + other_matrix[i + regions*num][j];
            }
            //cout << total << "\n";
            for (int j = 0; j < 32; j++) {
                aux = other_matrix[i + regions*num][j];
                other_normalized_matrix[i + regions*num][j] = aux/(total);
            }
            total = 0;
        }
    }

    for (int num = 0; num < 3; num++) {
        for (int i = 0; i < regions; i++) {
            for (int j = 0; j < 32; j++) {
                total = total + color_matrix[i + regions*num][j];
            }
            //cout << total << "\n";
            for (int j = 0; j < 32; j++) {
                aux = color_matrix[i + regions*num][j];
                color_normalized_matrix[i + regions*num][j] = aux/(total);
            }
            total = 0;
        }
    }

}

void generateArff(String name, char alg) {

    vl_uint32* segmentation;
    String arff_path = getUserDir() + name;
    String arff_name;
    String other_arff_name;
    String normalized_arff_name;
    Mat rmat;
    Mat mat;
    Mat aux_mat;
    String img_path;

    //SPECIFY THE RIGHT IMAGE FOLDER
    String path = "/Imgs/test1/image.orig/";

    int** matrix = NULL;
    int** other_matrix = NULL;
    int** color_matrix = NULL;
    float** normalized_matrix = NULL;
    float** other_normalized_matrix = NULL;
    float** color_normalized_matrix = NULL;
    int* supervisionedVector;
    int regions;
    bool active = false;
    char a[6];
    char b[5];

    for (int times = 0; times < 1; times ++) {

        arff_name = arff_path + "dual_manual_80_60_10000_last.arff";
        other_arff_name = arff_path + "single_color_manual_80_60_10000_last.arff";
        normalized_arff_name = arff_path + "dual_normalized_80_60_10000_last.arff";

        ofstream arff;
        ofstream other_arff;
        ofstream normalized_arff;

        arff.open(arff_name);
        other_arff.open(other_arff_name);
        normalized_arff.open(normalized_arff_name);

        //Choose which algorithm will run
        switch (alg) {
        case '1':
            for(int index = 111; index < 1000; index++) {
                itoa(index,a,10);
                img_path = getUserDir() + path + a + ".jpg";
                rmat = imread(img_path,CV_LOAD_IMAGE_COLOR);

                //GaussianBlur(rmat, mat, Size(5,5), 2, 2);
                //pyrMeanShiftFiltering(rmat, mat, 20, 45, 3);
                //bilateralFilter(rmat,mat,9,75,75);
                cvtColor(rmat,mat,CV_BGR2Lab);

                segmentation = new vl_uint32[mat.rows*mat.cols];

                //distanceTransform(mat);

                regions = generateSlicColor(mat,segmentation,false);

                //First iteration of the program will generate attribute lines based on regions
                //Will also allocate matrix space and free at the end of the program
                if (active == false) {
                    matrix = allocateMatrix(regions);
                    other_matrix = allocateMatrix(regions);
                    arff << "@relation " << alg << " \n";
                    other_arff << "@relation " << alg << " \n";
                    for (int i = 0; i < regions; i++) {
                        for (int k = 0; k < 32; k++) {
                            arff << "@attribute alg_" << alg << "_r_" << i << "_" << k << " numeric\n";
                            other_arff << "@attribute alg_" << alg << "_r_" << i << "_" << k << " numeric\n";
                        }
                    }
                    for (int i = 0; i < regions; i++) {
                        for (int k = 0; k < 32; k++) {
                            arff << "@attribute alg_" << alg << "_g_" << i << "_" << k << " numeric\n";
                            other_arff << "@attribute alg_" << alg << "_g_" << i << "_" << k << " numeric\n";
                        }
                    }
                    for (int i = 0; i < regions; i++) {
                        for (int k = 0; k < 32; k++) {
                            arff << "@attribute alg_" << alg << "_b_" << i << "_" << k << " numeric\n";
                            other_arff << "@attribute alg_" << alg << "_b_" << i << "_" << k << " numeric\n";
                        }
                    }
                    arff << "@attribute class {indio,praia,monumento,onibus,dinossauro,elefante,flor,cavalo,montanha,comida}\n@data ";
                    other_arff << "@attribute class {indio,praia,monumento,onibus,dinossauro,elefante,flor,cavalo,montanha,comida}\n@data ";
                    active = true;
                }
                cout << "Iterating number " << index << "\n";
                //Generate matrix to save to file, using image array, number of regions, segmentation labels and matrix pointer
                //aliveRGBMatrix(mat,regions,segmentation,matrix,other_matrix,NULL,NULL);

                //Saving in arff file
                for (int i = 0; i < regions*3; i++) {
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i][j] << ",";
                        other_arff << other_matrix[i][j] << ",";
                    }
                }

                arff << imageClass(index) << "\n";
                other_arff << imageClass(index) << "\n";

                //Free space
                free(segmentation);
            }
            break;
        case '2':
            for(int index = 0; index < 1000; index = index + 10) {
                itoa(index,a,10);
                img_path = getUserDir() + path + a + ".jpg";
                rmat = imread(img_path,CV_LOAD_IMAGE_COLOR);

                //GaussianBlur(rmat, mat, Size(5,5), 2, 2);
                //pyrMeanShiftFiltering(rmat, mat, 20, 45, 3);
                //bilateralFilter(rmat,mat,9,75,75);
                cvtColor(rmat,mat,CV_BGR2Lab);

                segmentation = new vl_uint32[mat.rows*mat.cols];

                //distanceTransform(mat);

                regions = generateSlicColor(mat,segmentation,false);

                //imshow("LAB image",mat);
                //waitKey(0);

                //First iteration of the program will generate attribute lines based on regions
                //Will also allocate matrix space and free at the end of the program
                if (active == false) {
                    matrix = allocateMatrix(regions);
                    arff << "@relation " << alg << " \n";
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_r_" << k << " numeric\n";
                    }
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_g_" << k << " numeric\n";
                    }
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_b_" << k << " numeric\n";
                    }
                    arff << "@attribute class {indio,praia,monumento,onibus,dinossauro,elefante,flor,cavalo,montanha,comida}\n@data ";
                    active = true;
                }
                cout << "Iterating number " << index << "\n";
                //Generate matrix to save to file, using image array, number of regions, segmentation labels and matrix pointer
                //aliveRGBMatrix(mat,regions,segmentation,matrix,NULL,NULL,NULL);

                //Saving in arff file
                for (int i = 0; i < regions; i++) {
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i][j] << ",";
                    }
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i+(regions*1)][j] << ",";
                    }
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i+(regions*2)][j] << ",";
                    }
                    arff << imageClass(index) << "\n";
                }

                //Free space
                free(segmentation);
            }
            break;
            //USE THIS ONE FOR SUPERPIXEL CLASSIFCATION
            //------------USE THIS------------
            //USE THIS ONE FOR SUPERPIXEL CLASSIFCATION
        case '3':
            //Specify the index range of the images
            for(int index = 600; index < 620; index++) {
                itoa(index,a,10);

                img_path = getUserDir() + path + a + ".jpg";
                aux_mat = imread(img_path,CV_LOAD_IMAGE_COLOR);
                cvtColor(aux_mat,rmat,CV_RGB2Lab);

                //rmat is used to help the user classify
                //mat is used by the program to extract
                mat = rmat.clone();

                segmentation = new vl_uint32[rmat.rows*rmat.cols];

                regions = generateSlicColor(rmat,segmentation,true);

                //First iteration of the program will generate attribute lines based on regions
                //Will also allocate matrix space and free at the end of the program
                if (active == false) {
                    matrix = allocateMatrix(regions);
                    other_matrix = allocateMatrix(regions);
                    color_matrix = allocateMatrix(regions);
                    normalized_matrix = allocateFloatMatrix(regions);
                    other_normalized_matrix = allocateFloatMatrix(regions);
                    color_normalized_matrix = allocateFloatMatrix(regions);
                    arff << "@relation " << alg << " \n";
                    other_arff << "@relation " << alg << " \n";
                    normalized_arff << "@relation " << alg << " \n";
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_r_ll_" << k << " numeric\n";
                        arff << "@attribute alg_" << alg << "_r_ld_" << k << " numeric\n";
                        normalized_arff << "@attribute alg_" << alg << "_r_ll_" << k << " numeric\n";
                        normalized_arff << "@attribute alg_" << alg << "_r_ld_" << k << " numeric\n";
                        other_arff << "@attribute alg_" << alg << "_r_" << k << " numeric\n";
                    }
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_g_ll_" << k << " numeric\n";
                        arff << "@attribute alg_" << alg << "_g_ld_" << k << " numeric\n";
                        normalized_arff << "@attribute alg_" << alg << "_g_ll_" << k << " numeric\n";
                        normalized_arff << "@attribute alg_" << alg << "_g_ld_" << k << " numeric\n";
                        other_arff << "@attribute alg_" << alg << "_g_" << k << " numeric\n";
                    }
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_b_ll_" << k << " numeric\n";
                        arff << "@attribute alg_" << alg << "_b_ld_" << k << " numeric\n";
                        normalized_arff << "@attribute alg_" << alg << "_b_ll_" << k << " numeric\n";
                        normalized_arff << "@attribute alg_" << alg << "_b_ld_" << k << " numeric\n";
                        other_arff << "@attribute alg_" << alg << "_b_" << k << " numeric\n";
                    }
                    //HERE YOU CAN ADD/REMOVE THE CLASSES
                    arff << "@attribute class {1,2}\n@data ";
                    other_arff << "@attribute class {1,2}\n@data ";
                    normalized_arff << "@attribute class {1,2}\n@data ";
                    active = true;
                }

                supervisionedVector = supervisionedClass(rmat,segmentation,regions);

                regions = generateSlicColor(mat,segmentation,false);

                cout << "Iterating number " << index << "\n";

                //Generate matrix to save to file, using image array, number of regions, segmentation labels and matrix pointer
                aliveRGBMatrix(mat,regions,segmentation,matrix,other_matrix,color_matrix,normalized_matrix,other_normalized_matrix,color_normalized_matrix);

                //Saving in arff file
                for (int i = 0; i < regions; i++) {
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i][j] << ",";
                        arff << other_matrix[i][j] << ",";
                        normalized_arff << normalized_matrix[i][j] << ",";
                        normalized_arff << other_normalized_matrix[i][j] << ",";
                        other_arff << color_normalized_matrix[i][j] << ",";
                    }
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i+(regions*1)][j] << ",";
                        arff << other_matrix[i+(regions*1)][j] << ",";
                        normalized_arff << normalized_matrix[i+(regions*1)][j] << ",";
                        normalized_arff << other_normalized_matrix[i+(regions*1)][j] << ",";
                        other_arff << color_normalized_matrix[i+(regions*1)][j] << ",";
                    }
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i+(regions*2)][j] << ",";
                        arff << other_matrix[i+(regions*2)][j] << ",";
                        normalized_arff << normalized_matrix[i+(regions*2)][j] << ",";
                        normalized_arff << other_normalized_matrix[i+(regions*2)][j] << ",";
                        other_arff << color_normalized_matrix[i+(regions*2)][j] << ",";
                    }
                    arff << supervisionedVector[i] << "\n";
                    other_arff << supervisionedVector[i] << "\n";
                    normalized_arff << supervisionedVector[i] << "\n";
                }

                //Free space
                free(segmentation);
                free(supervisionedVector);
            }
            break;
        case '4':
            for(int index = 0; index < 1000; index = index + 10) {
                itoa(index,a,10);
                img_path = getUserDir() + path + a + ".jpg";
                rmat = imread(img_path,CV_LOAD_IMAGE_COLOR);

                //GaussianBlur(rmat, mat, Size(5,5), 2, 2);
                //pyrMeanShiftFiltering(rmat, mat, 20, 45, 3);
                //bilateralFilter(rmat,mat,9,75,75);
                cvtColor(rmat,mat,CV_BGR2Lab);

                segmentation = new vl_uint32[mat.rows*mat.cols];

                //distanceTransform(mat);

                regions = generateSlicColor(mat,segmentation,false);

                //imshow("LAB image",mat);
                //waitKey(0);

                //First iteration of the program will generate attribute lines based on regions
                //Will also allocate matrix space and free at the end of the program
                if (active == false) {
                    matrix = allocateMatrix(regions);
                    arff << "@relation " << alg << " \n";
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_r_" << k << " numeric\n";
                    }
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_g_" << k << " numeric\n";
                    }
                    for (int k = 0; k < 32; k++) {
                        arff << "@attribute alg_" << alg << "_b_" << k << " numeric\n";
                    }
                    arff << "@data\n";
                    active = true;
                }
                cout << "Iterating number " << index << "\n";
                //Generate matrix to save to file, using image array, number of regions, segmentation labels and matrix pointer
                //aliveRGBMatrix(mat,regions,segmentation,matrix,NULL,NULL,NULL);

                //Saving in arff file
                for (int i = 0; i < regions; i++) {
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i][j] << ",";
                    }
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i+(regions*1)][j] << ",";
                    }
                    for (int j = 0; j < 32; j++) {
                        arff << matrix[i+(regions*2)][j] << ",";
                    }
                    arff << "\n";
                }

                //Free space
                free(segmentation);
            }
            break;
        default:
            cout << "No algorithm found.";
        }
        freeMatrix(matrix,regions);
        freeMatrix(other_matrix,regions);
        freeFloatMatrix(normalized_matrix,regions);
        freeFloatMatrix(color_normalized_matrix,regions);
        arff.close();
        other_arff.close();
        normalized_arff.close();
        active = false;
    }
}

int main(int argc, char *argv[])
{
    //Using method 3 that classifies superpixels manually
    //20 flower images classifying into 2 classes (1-flower/2-background).
    generateArff("/Arquivos .arff/",'3');
}
