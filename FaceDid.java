package com.pkg;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.DetectionModel;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class FaceDid {
	
	int absoluteFaceSize = 0;
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		new FaceDid().detectAndSave(Imgcodecs.imread("chacha.jpg"));
	}
	
	private void detectAndSave(Mat frame) {
		
		
		
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		//convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		if(absoluteFaceSize==0) {
			int height = grayFrame.height();
			if(Math.round(height * 0.2f) > 0) {
				absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		//detect faces
		CascadeClassifier faceCascade = new CascadeClassifier();
		faceCascade.load("haarcascade_frontalface_alt.xml");
		faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0|Objdetect.CASCADE_SCALE_IMAGE, 
				new Size(absoluteFaceSize,absoluteFaceSize), new Size()	);
		
		Rect[] faceArray = faces.toArray();
		System.out.println(faceArray.length);
		for(int i = 0; i < faceArray.length; i++) {
			Imgproc.rectangle(grayFrame, faceArray[i], new Scalar(0,0,255), 3);
		}
		Imgcodecs.imwrite("out.jpg",grayFrame);
	}

}
