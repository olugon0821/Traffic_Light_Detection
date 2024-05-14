package com.example.project.tcpip;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;
import java.rmi.RemoteException;

public class AppMain extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws IOException {

        try {
            final FXMLLoader loader = new FXMLLoader();
            final URL resource = this.getClass().getResource("app-main-controller.fxml");
            loader.setLocation(resource);
            final Parent parent_root = loader.load();
            final Scene scene = new Scene(parent_root);
            primaryStage.setTitle("ARDUINO TCP");
            primaryStage.setMaxWidth(800.0);
            primaryStage.setMaxHeight(1000.0);
            primaryStage.setResizable(false);
            primaryStage.setScene(scene);
            primaryStage.show();
        }catch (IOException e){
            throw new RemoteException();
        }
    }
}
