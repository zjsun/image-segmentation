package it.polito.teaching.cv;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class ImageSegmentation extends Application {
    /**
     * The main class for a JavaFX application. It creates and handle the main
     * window with its resources (style, graphics, etc.).
     * <p>
     * This application apply the Canny filter to the camera video stream or try
     * to remove a uniform background with the erosion and dilation operators.
     *
     * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
     * @version 1.1 (2015-11-24)
     * @since 1.0 (2013-12-20)
     */
    @Override
    public void start(Stage primaryStage) {
        try {
            // load the FXML resource
            BorderPane root = (BorderPane) FXMLLoader.load(getClass().getResource("/ImageSeg.fxml"));
            // set a whitesmoke background
            root.setStyle("-fx-background-color: whitesmoke;");
            // create and style a scene
            Scene scene = new Scene(root, 800, 600);
            scene.getStylesheets().add(getClass().getResource("/application.css").toExternalForm());
            // create the stage with the given title and the previously created
            // scene
            primaryStage.setTitle("Image Segmentation");
            primaryStage.setScene(scene);
            // show the GUI
            primaryStage.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
