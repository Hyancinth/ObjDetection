from yolo3 import*
import cv2
import tensorflow as tf
import config

def main():
    try:
        #load yolo3 model
        modelPath = config.modelPath
        yolo3 = tf.keras.models.load_model(modelPath, compile = False)

        #setup and start webcam 
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

        while True:
            ret, img = cam.read()
            img_h, img_w = cam.get(4), cam.get(3)
            newImg = preprocess_input(img, img_h, img_w)

            yolos = yolo3.predict(newImg) 
            boxes = []
            
            for i in range(len(yolos)):
                boxes += decode_netout(yolos[i][0], config.anchors[i], config.obj_thresh, config.nms_thresh, config.net_h, config.net_w)
            
            correct_yolo_boxes(boxes, img_h, img_w, config.net_h, config.net_w)

            do_nms(boxes, config.nms_thresh)
            draw_boxes(img, boxes, config.labels, config.obj_thresh)
            cv2.imshow("", img)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break   
        
        cam.release()
        cv2.destroyAllWindows()

    except FileNotFoundError:
        #file not found, make model, load weights and save 
        #maybe delete main in yolo3.py cause it's now redundant?
        print("\n[In Progress] Making Model")
        Model = make_yolov3_model()
        print("\n[Success] Model made")


        Model.compile(optimizer="adam", loss="mean_squared_error")

        # load the weights trained on COCO into the model
        weights_path = config.weightPath
        weight_reader = WeightReader(weights_path)
        weight_reader.load_weights(Model)


        #save model 
        print("\n[In Progress] Saving Model")
        Model.save("yolo3.h5", save_format = "h5")
        print("\n[Success] Done saving")

        #call main and try again
        main()

main()
