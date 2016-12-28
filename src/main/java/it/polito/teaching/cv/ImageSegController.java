package it.polito.teaching.cv;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_videoio;

import java.io.ByteArrayInputStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.imencode;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the image segmentation process.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.5 (2015-11-24)
 * @since 1.0 (2013-12-20)
 */
public class ImageSegController {

    // FXML buttons
    @FXML
    private Button cameraButton;
    // the FXML area for showing the current frame
    @FXML
    private ImageView originalFrame;
    // checkbox for enabling/disabling Canny
    @FXML
    private CheckBox canny;
    // canny threshold value
    @FXML
    private Slider threshold;
    // checkbox for enabling/disabling background removal
    @FXML
    private CheckBox dilateErode;
    // inverse the threshold value for background removal
    @FXML
    private CheckBox inverse;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that performs the video capture
    private opencv_videoio.VideoCapture capture = new opencv_videoio.VideoCapture();
    // a flag to change the button behavior
    private boolean cameraActive;

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera() {
        // set a fixed width for the frame
        originalFrame.setFitWidth(380);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);

        if (!this.cameraActive) {
            // disable setting checkboxes
            this.canny.setDisable(true);
            this.dilateErode.setDisable(true);

            // start the video capture
            this.capture.open(0);

            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run() {
                        Image imageToShow = grabFrame();
                        originalFrame.setImage(imageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");
            // enable setting checkboxes
            this.canny.setDisable(false);
            this.dilateErode.setDisable(false);
            // stop the timer
            try {
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log the exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }

            // release the camera
            this.capture.release();
            // clean the frame
            this.originalFrame.setImage(null);
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Image grabFrame() {
        // init everything
        Image imageToShow = null;
        opencv_core.Mat frame = new opencv_core.Mat();

        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    // handle edge detection
                    if (this.canny.isSelected()) {
                        frame = this.doCanny(frame);
                    }
                    // foreground detection
                    else if (this.dilateErode.isSelected()) {
                        frame = this.doBackgroundRemoval(frame);
                    }

                    // convert the Mat object (OpenCV) to Image (JavaFX)
                    imageToShow = mat2Image(frame);
                }

            } catch (Exception e) {
                // log the (full) error
                System.err.print("ERROR");
                e.printStackTrace();
            }
        }

        return imageToShow;
    }

    /**
     * Perform the operations needed for removing a uniform background
     *
     * @param frame the current frame
     * @return an image with only foreground objects
     */
    private opencv_core.Mat doBackgroundRemoval(opencv_core.Mat frame) {
        // init
        opencv_core.Mat hsvImg = new opencv_core.Mat();
        opencv_core.MatVector hsvPlanes = new opencv_core.MatVector();
        opencv_core.Mat thresholdImg = new opencv_core.Mat();

        int thresh_type = THRESH_BINARY_INV;
        if (this.inverse.isSelected())
            thresh_type = THRESH_BINARY;

        // threshold the image with the average hue value
        hsvImg.create(frame.size(), CV_8U);
        cvtColor(frame, hsvImg, COLOR_BGR2HSV);
        split(hsvImg, hsvPlanes);

        // get the average hue value of the image
        double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));

        threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, thresh_type);

        blur(thresholdImg, thresholdImg, new opencv_core.Size(5, 5));

        // dilate to fill gaps, erode to smooth edges
        dilate(thresholdImg, thresholdImg, new opencv_core.Mat(), new opencv_core.Point(-1, -1), 1, BORDER_CONSTANT, opencv_core.Scalar.all(0));
        erode(thresholdImg, thresholdImg, new opencv_core.Mat(), new opencv_core.Point(-1, -1), 3, BORDER_CONSTANT, opencv_core.Scalar.all(0));

        threshold(thresholdImg, thresholdImg, threshValue, 179.0, THRESH_BINARY);

        // create the new image
        opencv_core.Mat foreground = new opencv_core.Mat(frame.size(), CV_8UC3, Scalar.all(255));
        frame.copyTo(foreground, thresholdImg);

        return foreground;
    }

    /**
     * Get the average hue value of the image starting from its Hue channel
     * histogram
     *
     * @param hsvImg    the current frame in HSV
     * @param hueValues the Hue component of the current frame
     * @return the average Hue value
     */
    private double getHistAverage(Mat hsvImg, Mat hueValues) {
        // init
        double average = 0.0;
        Mat hist_hue = new Mat();
        // 0-180: range of Hue values
        int[] histSize = new int[]{180};
        MatVector hue = new MatVector();
        hue.put(new Mat[]{hueValues});

        // compute the histogram
        calcHist(hue, new int[]{0}, new Mat(), hist_hue, histSize, new float[]{0, 179});
        FloatIndexer hist_hue_idx = hist_hue.createIndexer();

        // get the average Hue value of the image
        // (sum(bin(h)*h))/(image-height*image-width)
        // -----------------
        // equivalent to get the hue of each pixel in the image, add them, and
        // divide for the image size (height and width)
        for (int h = 0; h < 180; h++) {
            // for each bin, get its value and multiply it for the corresponding
            // hue
            average += (hist_hue_idx.get(h, 0, 0) * h);
        }

        // return the average hue of the image
        return average = average / hsvImg.size().height() / hsvImg.size().width();
    }

    /**
     * Apply Canny
     *
     * @param frame the current frame
     * @return an image elaborated with Canny
     */
    private Mat doCanny(Mat frame) {
        // init
        Mat grayImage = new Mat();
        Mat detectedEdges = new Mat();

        // convert to grayscale
        cvtColor(frame, grayImage, COLOR_BGR2GRAY);

        // reduce noise with a 3x3 kernel
        blur(grayImage, detectedEdges, new Size(3, 3));

        // canny detector, with ratio of lower:upper threshold of 3:1
        Canny(detectedEdges, detectedEdges, this.threshold.getValue(), this.threshold.getValue() * 3);

        // using Canny's output as a mask, display the result
        Mat dest = new Mat();
        frame.copyTo(dest, detectedEdges);

        return dest;
    }

    /**
     * Action triggered when the Canny checkbox is selected
     */
    @FXML
    protected void cannySelected() {
        // check whether the other checkbox is selected and deselect it
        if (this.dilateErode.isSelected()) {
            this.dilateErode.setSelected(false);
            this.inverse.setDisable(true);
        }

        // enable the threshold slider
        if (this.canny.isSelected())
            this.threshold.setDisable(false);
        else
            this.threshold.setDisable(true);

        // now the capture can start
        this.cameraButton.setDisable(false);
    }

    /**
     * Action triggered when the "background removal" checkbox is selected
     */
    @FXML
    protected void dilateErodeSelected() {
        // check whether the canny checkbox is selected, deselect it and disable
        // its slider
        if (this.canny.isSelected()) {
            this.canny.setSelected(false);
            this.threshold.setDisable(true);
        }

        if (this.dilateErode.isSelected())
            this.inverse.setDisable(false);
        else
            this.inverse.setDisable(true);

        // now the capture can start
        this.cameraButton.setDisable(false);
    }

    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    private Image mat2Image(Mat frame) {
        // create a temporary buffer
        byte[] buffer = new byte[frame.cols() * frame.rows() * frame.channels()];
        // encode the frame in the buffer, according to the PNG format
        imencode(".png", frame, buffer);

        // build and return an Image created from the image encoded in the
        // buffer
        return new Image(new ByteArrayInputStream(buffer));
    }

}
