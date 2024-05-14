package com.example.project.tcpip;

import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.paint.Paint;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.fxml.FXML;
import javafx.scene.control.TextArea;

import java.io.IOException;
import javax.swing.*;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ResourceBundle;

public class AppMainController implements Initializable {

    public Circle red_led;
    public Circle blue_led;
    public Circle yellow_led;
    public Circle left_led;
    public Circle green_led;

    private final Socket socket;

    public AppMainController(){
        this.socket = new Socket();
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        try {
            this.socket.connect(new InetSocketAddress("192.168.0.8", Integer.parseInt("9999")));
            this.received_data_from_server(new ActionEvent());
        } catch (IOException e) {
//            throw new RuntimeException(e);
            System.out.printf("%s\r\n",e.getMessage());
        }
    }


    private void received_data_from_server(ActionEvent event) {
        Thread thread_of_receiving = new Thread(() -> {
            while (true) {
                try {
                    final InputStream inputStream = this.socket.getInputStream();
                    byte[] bytes_data = new byte[512];
                    final int read_byte_count = inputStream.read(bytes_data);
                    final String serial_input_data = new String(bytes_data, 0, read_byte_count, StandardCharsets.UTF_8);

                    System.out.printf("%s\r\n",serial_input_data);

                    Platform.runLater(() -> {
                        if (serial_input_data.equals("red\n")) {
                            red_led.setFill(Paint.valueOf("#FF0000"));
                            green_led.setFill(Paint.valueOf("#000000"));
                            yellow_led.setFill(Paint.valueOf("#000000"));
                        } else if(serial_input_data.equals("green\n")) {
                            red_led.setFill(Paint.valueOf("#000000"));
                            green_led.setFill(Paint.valueOf("#00FF00"));
                            yellow_led.setFill(Paint.valueOf("#000000"));
                        } else if (serial_input_data.equals("yellow\n")) {
                            red_led.setFill(Paint.valueOf("#000000"));
                            green_led.setFill(Paint.valueOf("#000000"));
                            yellow_led.setFill(Paint.valueOf("#FFFF00"));
                        }
                    });

                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        thread_of_receiving.start();
    }
}