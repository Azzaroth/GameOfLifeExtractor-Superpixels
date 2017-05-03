QT += core
QT -= gui

CONFIG += c++11

TARGET = AlgoritmoGeraArff
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    slic.c \
    generic.c \
    host.c \
    random.c

INCLUDEPATH += C:\\opencv\\compilado\\install\\include
LIBS += -LC:\\opencv\\compilado\\lib \
-lopencv_core2410.dll \
-lopencv_highgui2410.dll \
-lopencv_imgproc2410.dll \

HEADERS += \

