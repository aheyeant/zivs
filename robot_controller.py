# coding=utf-8
import argparse
import random
import sys
import almath
import cv2
import imutils
import naoqi
import numpy as np
import time


class Controller(object):
    # if images_name is not none, get images from memory, else get photo from camera
    # note: images_name = (original, [part, name])
    def __init__(self, robot_ip, port=9559, count_parts=(4, 3)):
        self.robot_ip = robot_ip
        self.port = port
        try:
            self.my_broker = naoqi.ALBroker("myBroker", "0.0.0.0", 0, robot_ip, port)
        except Exception as exception:
            print(exception)
            print("FATAL ERROR. Problem 'ALBroker'. Finish program")
            return
        print "ALBroker - OK"
        global detector

        try:
            detector = ReactToTouch("ReactToTouch")
        except Exception as exception:
            print(exception)
            print("FATAL ERROR. Problem 'ReactToTouch'. Finish program")
            return
        print "ReactToTouch - OK"
        if robot_ip is None:
            print("incorrect robot ip")
            raise ValueError
        print "robot ip - OK"
        try:
            self.robotPosture = naoqi.ALProxy("ALRobotPosture", robot_ip, port)
        except Exception as exception:
            print(exception)
            print("FATAL ERROR. Problem: 'ALRobotPosture'. Finish program")
            return
        print "ALRobotPosture - OK"
        try:
            self.motion = naoqi.ALProxy("ALMotion", robot_ip, port)
        except Exception as exception:
            print(exception)
            print("FATAL ERROR. Problem: 'ALMotion'. Finish program")
            return
        print "ALMotion - OK"
        try:
            self.speech = naoqi.ALProxy("ALTextToSpeech", robot_ip, port)
            self.speech.setLanguage("Czech")
        except Exception as exception:
            print(exception)
            print("ERROR. Problem: 'ALTextToSpeech'. Program continue without sound")
            self.speech = None
        print "ALTextToSpeech - OK"
        try:
            self.video_device = naoqi.ALProxy("ALVideoDevice", robot_ip, port)
        except Exception as exception:
            print(exception)
            print("FATAL ERROR. Problem: 'ALVideoDevice'. Finish program")
            return
        print "ALVideoDevice - OK"
        self.original_photo = []
        self.parts_photo = []
        self.count_parts = count_parts
        self.builder = PuzzleBuilder(count_parts, "RGB", 20, 8, 0)
        self.numToText = {1: "jedna",
                          2: "dva",
                          3: "tři",
                          4: "čtyři",
                          5: "pět",
                          6: "šest",
                          7: "sedm",
                          8: "osm",
                          9: "devět",
                          10: "deset",
                          11: "jedenáct",
                          12: "dvanáct"}
        self.directionToText = {"top": "",
                                "right": " a otáč jeho na 270 stupňů",
                                "bottom": " a otáč jeho na 180 stupňů",
                                "left": " a otáč jeho na 90 stupňů"}

    def startEpisode(self, tested=0):
        self.standUp()
        while self.robotPosture.getPosture() != "Stand":
            pass
        self.sayMessage("Ahoj")
        time.sleep(0.2)
        self.watchToFloor()
        time.sleep(0.2)
        self.sayMessage("Můžeme začít")
        time.sleep(0.2)
        self.waitAnswerFromSensors()
        image_original = self.getPhoto(3, 10, tested)
        self.builder.readAndSliceOriginalImage(image_original, tested)
        self.sayMessage("ok")
        for i in range(self.count_parts[0] * self.count_parts[1]):
            self.waitAnswerFromSensors()
            image_part = self.getPhoto(3, 10, tested)
            self.builder.readOnePuzzle(image_part, 100)
            self.builder.assemblyPuzzleFromOneSlices(tested)
            item = self.builder.result[len(self.builder.result) - 1]
            self.sayAnswer(item)
        self.sayMessage("bue")

    # input: resolution = {3 -> 1280 x 960; 2 -> 640 x 480}
    # input: cut_delta is nim in percents, how pixels cut from one dir
    #   _____________
    #   |           |
    #   |           |
    #   |<->|       |
    #   |           |
    #   |           |
    #   |           |
    #   -------------
    def getPhoto(self, resolution=3, cut_delta=10, show=0):
        self.video_device.unsubscribeAllInstance("cam")
        cam_id = 1
        if resolution == 2:
            size = (480, 640)
        else:
            size = (960, 1280)
        color_space = 13  # format RGB
        fps = 30
        cam = self.video_device.subscribeCamera("cam", cam_id, resolution, color_space, fps)  # zabrani kamery
        image = self.video_device.getImageRemote("cam")  # porizeni snimku
        im = image[6]
        ret = np.fromstring(im, np.uint8)  # konverze na cisla
        ret = ret.reshape(size[0], size[1], 3)
        he_start = size[0] - (((100 - cut_delta) * size[0]) / 100)
        wi_start = size[1] - (((100 - cut_delta) * size[1]) / 100)
        ret = ret[he_start: size[0] - he_start: 1, wi_start: size[1] - he_start: 1]
        self.video_device.unsubscribe("cam")
        if show:
            cv2.imshow("camera", ret)  # zobrazeni vysledneho obrazu
            cv2.waitKey(0)
        return ret

    @staticmethod
    def waitAnswerFromSensors():
        detector.flag = 0
        while not detector.flag:
            pass

    # note: control in try-except bloc
    def standUp(self):
        self.robotPosture.goToPosture("Stand", 1.0)

    def watchToFloor(self):
        self.motion.setStiffnesses("Head", 1.0)
        self.motion.setAngles("HeadYaw", 0, 0.1)
        self.motion.setAngles("HeadPitch", 29.5 * almath.TO_RAD, 0.1)
        self.motion.setStiffnesses("Head", 0)

    # note: control in try-except bloc
    # note: control in try-except bloc
    def sayMessage(self, message=None):
        if self.speech is None or message is None:
            return
        else:
            self.speech.say(str(message))

    def sayAnswer(self, status_image):
        self.sayMessage("dej tento puzzle na pozice " + str(status_image[0] + 1) + self.directionToText[status_image[1]])

    # exit: keyboard interrupt
    def testAnswer(self):
        try:
            while not None:
                print "cislo 1 - 12"
                num = input()
                print "t(top), r(right), b(bottom), l(left)"
                str_dir = input()
                if str_dir == "t":
                    str_dir = "top"
                if str_dir == "r":
                    str_dir = "right"
                if str_dir == "l":
                    str_dir = "left"
                if str_dir == "b":
                    str_dir = "bottom"
                self.sayAnswer((int(num), str_dir))
        except KeyboardInterrupt:
            print "finish program"
            return

    # exit: print 'exit' or keyboard interrupt

    # exit: print 'exit' or keyboard interrupt
    def testSay(self):
        back = 0
        try:
            while not back:
                print "print text or 'exit' for return"
                in_text = input(str)
                if in_text == "exit":
                    back = 1
                self.sayMessage(in_text)
        except KeyboardInterrupt:
            print "finish program"
            return

    # exit: print 'exit' or keyboard interrupt
    # print num in percent
    def testGetPhoto(self):
        back = 0
        try:
            while not back:
                in_case = input()
                if in_case == "exit":
                    print "finish program"
                    return
                num = None
                if str.isdigit(in_case):
                    num = int(in_case)
                original_photo = self.getPhoto(3, 0, 0)
                if num is not None and 0 >= num >= 50:
                    num = None
                if num is None:
                    num = 0
                cut = self.getPhoto(3, num, 0)
                cv2.imshow("original", original_photo)
                cv2.imshow("cut " + str(num) + "%", cut)
                cv2.waitKey(100)
        except KeyboardInterrupt:
            print "finish program"
            return

    # exit: keyboard interrupt
    @staticmethod
    def testSensors():
        try:
            while 1:
                print detector.flag
                time.sleep(0.1)
        except KeyboardInterrupt:
            print "finish program"

    def testHeadMotion(self):
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("yaw", "test", -119.5, 119.5, lambda x: x)
        cv2.createTrackbar("pitch", "test", -38.5, 29.5, lambda x: x)
        self.motion.setStiffnesses("Head", 1.0)
        try:
            while not None:
                yaw = cv2.getTrackbarPos("yaw", "test")
                pitch = cv2.getTrackbarPos("pitch", "test")
                self.motion.setAngles("HeadYaw", yaw * almath.TO_RAD, 0.1)
                time.sleep(0.5)
                self.motion.setAngles("HeadPitch", pitch * almath.TO_RAD, 0.1)
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.motion.setStiffnesses("Head", 0)
        print "finish program"
        return


class ReactToTouch(naoqi.ALModule):
    def __init__(self, name):
        naoqi.ALModule.__init__(self, name)
        self.memory = naoqi.ALProxy("ALMemory")
        self.memory.subscribeToEvent("MiddleTactilTouched", "ReactToTouch", "onTouched")
        self.flag = 0

    def onTouched(self):
        self.memory.unsubscribeToEvent("MiddleTactilTouched", "ReactToTouch")
        self.flag = 1
        self.memory.subscribeToEvent("MiddleTactilTouched", "ReactToTouch", "onTouched")


class PuzzleBuilder(object):
    # URI of original puzzle
    # array URIs where are parts from original puzzle
    # perts_count -> (to height, to width)
    # colour_scheme = {RGB, BGR}
    # shuffle_init = 0(off), 1(random, no write), 2(random, write)
    # self.result -> array(i, j, direction) => on the position _i_ may be parts _j_(now rotate _dir_)
    def __init__(self, parts_count=(4, 3), colour_scheme="RGB", contours_eps=20,
                 sqrt_start_hash_size=8, shuffle_init=0):
        self.original_name = None
        self.parts_name = None
        self.parts_count = parts_count
        self.slices_full_image = []
        self.slices_part_image = []
        self.colour_scheme = colour_scheme
        self.contours_eps = contours_eps
        self.sqrt_start_hash_size = sqrt_start_hash_size
        self.shuffle_init = shuffle_init
        self.original_slice_size = ()
        self.result = []
        self.hash_table_original_slices = None
        if self.colour_scheme != "RGB" and self.colour_scheme != "BGR":
            raise ValueError

    def start(self, original_name, parts_name, tested=0):
        self.original_name = original_name
        self.parts_name = parts_name
        if self.original_name is None or self.parts_name is None:
            raise ValueError
        if len(self.parts_name) != (self.parts_count[0] * self.parts_count[1]):
            raise ValueError

        try:
            self.readAndSliceOriginalImage(tested)
            self.readPartsOfPuzzle(120)
            if tested:
                self.testTwoSlicesStructure()
                self.imageFromSlice(self.slices_part_image)
                self.testHashStructure()
            self.assemblyPuzzleFromAllSlices(tested=1)
            self.showDifference()
        except Exception as exception:
            print(exception)
            raise exception

    # images = (original, [parts])
    def startWithOneSlice(self, images=None, tested=0):
        self.shuffle_init = 0
        try:
            if images is None:
                image = None
            else:
                image = images[0]
            self.readAndSliceOriginalImage(image, 80, tested)
            if images is None:
                image = None
            else:
                image = images[1]
            for i in image:
                self.readOnePuzzle(i, 100, tested)
                self.assemblyPuzzleFromOneSlices(tested=tested)
                print (self.result)

            # if tested:
                # self.testTwoSlicesStructure()
                # self.imageFromSlice(self.slices_part_image)
                # self.testHashStructure()
            # self.showDifference()
        except Exception as exception:
            print(exception)
            raise exception

    def readAndSliceOriginalImage(self, image=None, colour_max=80, tested=0):
        try:
            if image is None:
                image = cv2.imread(self.original_name)
            if tested:
                cv2.imshow("original_photo", image)
                cv2.waitKey(0)
            new_image = self.findRect(image, colour_max, 0, 10)
            if tested:
                cv2.imshow("parse original photo", new_image)
                cv2.waitKey(0)
            self.slices_full_image = self.sliceImage(new_image)
            self.original_slice_size = self.slices_full_image[0]
            self.hash_table_original_slices = self.getHashForSlices(self.slices_full_image, "top")
            self.slices_part_image = [(None, None), self.slices_full_image[1]]
        except Exception as exception:
            print(exception)
            raise exception

    def readPartsOfPuzzle(self, colour_max=100):
        self.slices_part_image = [(None, None), self.slices_full_image[1]]
        for i in range(self.slices_full_image[1]):
            im = cv2.imread(self.parts_name[i])
            new_im = self.findRect(im, colour_max, 0, None)
            self.slices_part_image.append(new_im)
        if self.shuffle_init == 1:
            self.shuffle("random", "no", None)
        if self.shuffle_init == 2:
            self.shuffle("random", "yes", None)

    def readOnePuzzle(self, image=None, colour_max=100, tested=0):
        if image is None:
            image = cv2.imread(self.parts_name[len(self.slices_part_image) - 2])
        new_im = self.findRect(image, colour_max, tested, None)
        self.slices_part_image.append(new_im)

    def findRect(self, image, colour_max=80, tested=0, delta=None):
        if image is None:
            return None
        test_range = {"min": np.array((0, 0, 0), np.uint8),
                      "max": np.array((colour_max, colour_max, colour_max), np.uint8)}
        thresh = cv2.inRange(image, test_range["min"], test_range["max"])
        self.showImage(image, tested)
        self.showImage(thresh, tested)

        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        if tested:
            print("rect => " + str(rect))
        if rect[2] == 0:
            angle = 0
        elif abs(rect[2]) < 45:
            angle = abs(rect[2])
        else:
            angle = 270 + abs(rect[2])
        rotate = imutils.rotate_bound(image, angle)
        self.showImage(rotate, tested)

        thresh = cv2.inRange(rotate, test_range["min"], test_range["max"])
        self.showImage(thresh, tested)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rotate, [box], -1, (0, 255, 0), 3)
        if tested:
            print(box)
            self.showImage(rotate, tested)
        ret = self.getImageFromContours(rotate, box, delta)
        self.showImage(ret, tested)
        return ret

    # return images array[(height, width), count ,arrays[(to_height * to_width)]...]
    #           1  2  3
    #           4  5  6
    #           7  8  9
    #          10 11 12
    def sliceImage(self, image):
        he_part = int(np.size(image, 0) / self.parts_count[0])
        wi_part = int(np.size(image, 1) / self.parts_count[1])
        ret = [(he_part, wi_part), self.parts_count[0] * self.parts_count[1]]
        for i in range(self.parts_count[1] * self.parts_count[0]):
            ret.append(np.zeros((he_part, wi_part, 3), np.uint8))
        for i in range(self.parts_count[0]):
            for j in range(self.parts_count[1]):
                pos_in_array = i * self.parts_count[1] + j + 1 + 1
                start_he = i * he_part
                start_wi = j * wi_part
                for ii in range(he_part):
                    for jj in range(wi_part):
                        ret[pos_in_array][ii][jj] = image[start_he + ii][start_wi + jj]
        return ret

    # contours may be type BOX
    def getImageFromContours(self, image, in_contours, delta=None):
        contours = []
        maximum = 0
        minimum = 10000
        iter_4 = 0
        iter_2 = 0
        for i in range(4):
            if maximum < (in_contours[i][0] + in_contours[i][1]):
                maximum = in_contours[i][0] + in_contours[i][1]
                iter_4 = i
            if minimum > (in_contours[i][0] + in_contours[i][1]):
                minimum = in_contours[i][0] + in_contours[i][1]
                iter_2 = i
        iter_3 = 0
        iter_1 = 0
        for i in range(4):
            if i != iter_4:
                if abs(in_contours[iter_4][0] - in_contours[i][0]) < self.contours_eps:
                    iter_3 = i
                    break
        for i in range(4):
            if i != iter_4:
                if abs(in_contours[iter_4][1] - in_contours[i][1]) < self.contours_eps:
                    iter_1 = i
                    break
        contours.append(in_contours[iter_1])
        contours.append(in_contours[iter_2])
        contours.append(in_contours[iter_3])
        contours.append(in_contours[iter_4])
        if delta is None:
            delta = int((contours[0][1] - contours[1][1]) * 2.5 / 70)
        he = contours[0][1] - contours[1][1] - 2 * delta
        wi = contours[2][0] - contours[1][0] - 2 * delta

        ret = np.zeros((he, wi, 3), np.uint8)
        point = contours[1]
        for i in range(he):
            for j in range(wi):
                ret[i][j] = image[point[1] + i + delta][point[0] + j + delta]
        return ret

    # input: slices structure -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    # note: hash_table -> (sqrt(hash_size) ,slices, array[dicts_of_hash]
    # note: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    # note: slices structure -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    # note: hammingDistanceAllDirectionToTop return -> (distance, direction)
    # note: puzzle_item -> (iterator of part puzzle, direction)
    def assemblyPuzzleFromAllSlices(self, tested=1):
        hash_table = self.getHashForSlices(self.slices_part_image, "all")
        for i in range(self.slices_full_image[1]):
            dict_of_hash = self.getHashForImage(self.slices_full_image[i + 2],
                                                "top")
            puzzle_item = self.findPuzzlePartWithHash(self.slices_full_image[i + 2],
                                                      dict_of_hash, hash_table)
            hash_table[2][puzzle_item[0]] = None
            self.result.append((i, puzzle_item[0], puzzle_item[1]))
            if tested:
                print("part of puzzle " + str(puzzle_item[0] + 1) + "(" + str(puzzle_item[1])
                      + ") have to be on the position " + str(i + 1))

    # input: slices structure -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    # note: hash_table -> (sqrt(hash_size) ,slices, array[dicts_of_hash]
    # note: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    # note: slices structure -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    # note: hammingDistanceAllDirectionToTop return -> (distance, direction)
    # note: puzzle_item -> (iterator of part puzzle, direction)
    def assemblyPuzzleFromOneSlices(self, tested=1):
        dict_of_hash = self.getHashForImage(self.slices_part_image[len(self.slices_part_image) - 1],
                                            "all")
        puzzle_item = self.findPuzzlePartWithHash(self.slices_part_image[len(self.slices_part_image) - 1],
                                                  dict_of_hash, self.hash_table_original_slices)
        self.hash_table_original_slices[2][puzzle_item[0]] = None
        self.result.append((puzzle_item[0], puzzle_item[1]))
        if tested:
            print("part of puzzle " + str(puzzle_item[0] + 1) + "(" + str(puzzle_item[1])
                  + ") have to be on the position " + str(1))

    # input: slices -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    #      : dir = {top, right, bottom, left, all}
    # return: hash_table -> (sqrt(hash_size) ,slices, array[dicts_of_hash]
    # note: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    def getHashForSlices(self, slices, direction="all"):
        ret = (self.sqrt_start_hash_size, slices, [])
        for i in range(slices[1]):
            ret[2].append(self.calculateHash(cv2.resize(slices[i + 2],
                                                        (self.sqrt_start_hash_size, self.sqrt_start_hash_size),
                                                        interpolation=cv2.INTER_AREA), direction))
        return ret

    # return: dict_of_hash
    # note: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    def getHashForImage(self, image, direction="all"):
        return self.calculateHash(cv2.resize(image, (self.sqrt_start_hash_size, self.sqrt_start_hash_size),
                                             interpolation=cv2.INTER_AREA), direction)

    # calculate hash for input image
    # input: image in size (size*size)
    #      : dir = {top, right, bottom, left, all}
    # return: (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    def calculateHash(self, image, direction="all"):
        if self.colour_scheme == "RGB":
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        median = int(0)
        for i in range(self.sqrt_start_hash_size):
            for j in range(self.sqrt_start_hash_size):
                median += gray_image[i][j]
        median = int(median / (self.sqrt_start_hash_size * self.sqrt_start_hash_size))
        ret = {"top": None, "right": None, "bottom": None, "left": None}

        if direction == "top":
            ret["top"] = ""
            for i in range(self.sqrt_start_hash_size):
                for j in range(self.sqrt_start_hash_size):
                    if gray_image[i][j] < median:
                        ret["top"] += "0"
                    else:
                        ret["top"] += "1"
            return self.sqrt_start_hash_size * self.sqrt_start_hash_size, ret

        hash_image = np.zeros((self.sqrt_start_hash_size, self.sqrt_start_hash_size, 1), np.uint8)
        for i in range(self.sqrt_start_hash_size):
            for j in range(self.sqrt_start_hash_size):
                if gray_image[i][j] < median:
                    hash_image[i][j] = 0
                else:
                    hash_image[i][j] = 1

        if direction == "all":
            ret["top"] = ""
            for i in range(self.sqrt_start_hash_size):
                for j in range(self.sqrt_start_hash_size):
                    if hash_image[i][j] == 1:
                        ret["top"] += "1"
                    else:
                        ret["top"] += "0"

            ret["right"] = ""
            for i in range(self.sqrt_start_hash_size - 1, -1, -1):
                for j in range(self.sqrt_start_hash_size):
                    if hash_image[j][i] == 1:
                        ret["right"] += "1"
                    else:
                        ret["right"] += "0"

            ret["bottom"] = ""
            for i in range(self.sqrt_start_hash_size - 1, -1, -1):
                for j in range(self.sqrt_start_hash_size - 1, -1, -1):
                    if hash_image[i][j] == 1:
                        ret["bottom"] += "1"
                    else:
                        ret["bottom"] += "0"

            ret["left"] = ""
            for i in range(self.sqrt_start_hash_size):
                for j in range(self.sqrt_start_hash_size - 1, -1, -1):
                    if hash_image[j][i] == 1:
                        ret["left"] += "1"
                    else:
                        ret["left"] += "0"
        else:
            if direction == "right":
                ret["right"] = ""
                for i in range(self.sqrt_start_hash_size - 1, -1, -1):
                    for j in range(self.sqrt_start_hash_size):
                        if hash_image[j][i] == 1:
                            ret["right"] += "1"
                        else:
                            ret["right"] += "0"

            if direction == "bottom":
                ret["bottom"] = ""
                for i in range(self.sqrt_start_hash_size - 1, -1, -1):
                    for j in range(self.sqrt_start_hash_size - 1, -1, -1):
                        if hash_image[i][j] == 1:
                            ret["bottom"] += "1"
                        else:
                            ret["bottom"] += "0"

            if direction == "left":
                ret["left"] = ""
                for i in range(self.sqrt_start_hash_size):
                    for j in range(self.sqrt_start_hash_size - 1, -1, -1):
                        if hash_image[j][i] == 1:
                            ret["left"] += "1"
                        else:
                            ret["left"] += "0"
        return self.sqrt_start_hash_size * self.sqrt_start_hash_size, ret

    # input: hash_table -> (sqrt(hash_size) ,slices, array[dicts_of_hash]
    #      : dict_of_hash
    # note: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    # note: slices structure -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    # note: hammingDistanceAllDirectionToTop return -> (distance, direction)
    # return: puzzle_item -> (iterator of part puzzle, direction)
    def findPuzzlePartWithHash(self, image, dict_from_hash, hash_table):
        first_result = []
        for i in range(hash_table[1][1]):
            if hash_table[2][i] is not None:
                first_result.append((i, self.hammingDistanceAllDirectionToTop(dict_from_hash, hash_table[2][i])))
        minimum = dict_from_hash[0] * dict_from_hash[0]
        for i in first_result:
            if minimum >= i[1][0]:
                minimum = i[1][0]
        second_result = []
        for i in first_result:
            if i[1][0] == minimum:
                second_result.append(i)
        if len(second_result) == 1:
            return second_result[0][0], second_result[0][1][1]

        iterators = []
        for i in second_result:
            iterators.append(i[0])
        size = dict_from_hash[0] + 1
        while not None:
            new_dict_from_hash = self.getHashForImage(image, "top")
            new_hash_images = []
            for i in iterators:
                new_hash_images.append((i, self.getHashForImage(hash_table[1][i + 2], "all")))
            first_result = []
            for i in new_hash_images:
                if new_hash_images[1] is not None:
                    first_result.append((i[0], self.hammingDistanceAllDirectionToTop(i[1], new_dict_from_hash)))
            minimum = size * size
            for i in first_result:
                if minimum >= i[1][0]:
                    minimum = i[1][0]
            second_result = []
            for i in first_result:
                if i[1][0] == minimum:
                    second_result.append(i)
            if len(second_result) == 1:
                return second_result[0][0], second_result[0][1][1]
            size += 1
            iterators.clear()
            for i in second_result:
                iterators.append(i[0])

    # input: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    # return: (distance, direction)
    def hammingDistanceAllDirectionToTop(self, dict_of_hash1, dict_of_hash2):
        distance = int(dict_of_hash1[0] * dict_of_hash1[0])
        iterator = int(0)
        test = [0, 0, 0, 0]
        test[0] = self.hammingDistance(dict_of_hash1, "top", dict_of_hash2, "top")
        test[1] = self.hammingDistance(dict_of_hash1, "right", dict_of_hash2, "top")
        test[2] = self.hammingDistance(dict_of_hash1, "bottom", dict_of_hash2, "top")
        test[3] = self.hammingDistance(dict_of_hash1, "left", dict_of_hash2, "top")
        for i in range(4):
            if test[i][0] < distance:
                iterator = i
                distance = test[i][0]
        return test[iterator]

    # input: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    # return: (distance, direction)
    @staticmethod
    def hammingDistance(dict_of_hash1, direction1, dict_of_hash2, direction2):
        ret = int(0)
        for i in range(dict_of_hash1[0]):
            if dict_of_hash1[1][direction1][i] != dict_of_hash2[1][direction2][i]:
                ret += 1
        return ret, direction1

    # note: mode = {random, iv}
    # note: on_window = {yes, no}
    # note: in iv min num is 0
    def shuffle(self, mod="random", on_window="no", iv=None):
        tmp = []
        for i in range(self.slices_part_image[1]):
            tmp.append((i, self.slices_part_image[i + 2]))
        if mod == "random":
            for i in range(self.slices_part_image[1]):
                iterator = random.randint(0, self.slices_part_image[1] - 1)
                pom = tmp[i]
                tmp[i] = tmp[iterator]
                tmp[iterator] = pom

            for i in range(self.slices_part_image[1]):
                self.slices_part_image[i + 2] = tmp[i][1]
            if on_window == "yes":
                print("shuffle")
                a = 0
                for i in tmp:
                    print(str(a + 1) + "->" + str(i[0] + 1))
                print("")
            return self.slices_part_image

        if mod == "iv":
            if iv is None:
                print("no iv")
                return self.slices_part_image
            if len(iv) != self.slices_part_image[1]:
                print("error len iv")
                return self.slices_part_image
            tmp = []
            for i in iv:
                tmp.append(self.slices_part_image[i + 2])
            for i in range(self.slices_part_image[1]):
                self.slices_part_image[i + 2] = tmp[i]
            if on_window == "yes":
                print("shuffle")
                a = 0
                for i in iv:
                    print(str(a + 1) + "->" + str(i[0] + 1))
                print("")
            return self.slices_part_image
        return self.slices_part_image

    # draw two window for full_slices and part_slices
    def testTwoSlicesStructure(self):
        cv2.namedWindow('slices_full_image', flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow('slices_part_image', flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("slices_full_image", 400, 400)
        cv2.resizeWindow("slices_part_image", 400, 400)
        cv2.createTrackbar('frame', 'slices_full_image', 1, self.slices_full_image[1], lambda x: x)
        cv2.createTrackbar('frame', 'slices_part_image', 1, self.slices_part_image[1], lambda x: x)
        while not None:
            fr1 = cv2.getTrackbarPos("frame", "slices_full_image")
            fr2 = cv2.getTrackbarPos("frame", "slices_part_image")
            if not fr1:
                fr1 = 1
            if not fr2:
                fr2 = 1
            cv2.imshow("slices_full_image", self.slices_full_image[fr1 + 1])
            cv2.imshow("slices_part_image", self.slices_part_image[fr2 + 1])
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyWindow("slices_full_image")
                cv2.destroyWindow("slices_part_image")
                break

    # note: hash_table -> (sqrt(hash_size) ,slices, array[dicts_of_hash]
    # note: dict_of_hash -> (hash_size, hash_dict)
    # note: dict {"top": hash or None,
    #             "left": hash or None,
    #             "bottom": hash or None,
    #             "right": hash or None}
    # note: slices structure -> array[(height, width), count ,arrays[(to_height * to_width)]...]
    def testHashStructure(self):
        hash_table_1 = self.getHashForSlices(self.slices_full_image)
        hash_table_2 = self.getHashForSlices(self.slices_part_image)
        cv2.namedWindow('slices_full_image', flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow('slices_part_image', flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("slices_full_image", hash_table_1[0] * 50, hash_table_1[0] * 50)
        cv2.resizeWindow("slices_part_image", hash_table_2[0] * 50, hash_table_2[0] * 50)
        if hash_table_1[2][0][1]["right"] is not None:
            dir1 = 4
        else:
            dir1 = 1

        if hash_table_2[2][0][1]["right"] is not None:
            dir2 = 4
        else:
            dir2 = 1
        cv2.createTrackbar('num', 'slices_full_image', 0, hash_table_1[1][1] - 1, lambda x: x)
        cv2.createTrackbar('dir', 'slices_full_image', 1, dir1, lambda x: x)
        cv2.createTrackbar('num', 'slices_part_image', 0, hash_table_2[1][1] - 1, lambda x: x)
        cv2.createTrackbar('dir', 'slices_part_image', 1, dir2, lambda x: x)
        while not None:
            num1 = cv2.getTrackbarPos("num", "slices_full_image")
            d1 = cv2.getTrackbarPos("dir", "slices_full_image")
            num2 = cv2.getTrackbarPos("num", "slices_part_image")
            d2 = cv2.getTrackbarPos("dir", "slices_part_image")
            if not d1:
                d1 = 1
            if not d2:
                d2 = 1
            if d1 == 1:
                d1 = "top"
            if d1 == 2:
                d1 = "right"
            if d1 == 3:
                d1 = "bottom"
            if d1 == 4:
                d1 = "left"
            if d2 == 1:
                d2 = "top"
            if d2 == 2:
                d2 = "right"
            if d2 == 3:
                d2 = "bottom"
            if d2 == 4:
                d2 = "left"
            image1 = np.zeros((hash_table_1[0], hash_table_1[0], 1), np.uint8)
            image2 = np.zeros((hash_table_2[0], hash_table_2[0], 1), np.uint8)
            for i in range(hash_table_1[0]):
                for j in range(hash_table_1[0]):
                    if hash_table_1[2][num1][1][d1][i * hash_table_1[0] + j] == "1":
                        image1[i][j] = 255
            for i in range(hash_table_2[0]):
                for j in range(hash_table_2[0]):
                    if hash_table_2[2][num2][1][d2][i * hash_table_2[0] + j] == "1":
                        image2[i][j] = 255

            cv2.imshow("slices_full_image", image1)
            cv2.imshow("slices_part_image", image2)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyWindow("slices_full_image")
                cv2.destroyWindow("slices_part_image`")
                break

    def imageFromSlice(self, slices):
        ret = np.zeros((self.parts_count[0] * self.original_slice_size[0],
                        self.parts_count[1] * self.original_slice_size[1], 3), np.uint8)

        for i in range(self.parts_count[0]):
            for j in range(self.parts_count[1]):
                num = i * self.parts_count[1] + j + 2
                start_he = i * self.original_slice_size[0]
                start_wi = j * self.original_slice_size[1]
                size = (self.original_slice_size[1], self.original_slice_size[0])
                image = cv2.resize(slices[num], size, interpolation=cv2.INTER_AREA)
                for ii in range(self.original_slice_size[0]):
                    for jj in range(self.original_slice_size[1]):
                        ret[start_he + ii][start_wi + jj] = image[ii][jj]
        return ret

    def finishImageBuild(self):
        directions = {"top": 0, "right": 270, "bottom": 180, "left": 90}
        tmp_slices = [self.original_slice_size, self.slices_part_image[1]]
        for i in self.result:
            tmp_slices.append(imutils.rotate_bound(self.slices_part_image[i[1] + 2], directions[i[2]]))
        return tmp_slices

    def showDifference(self):
        cv2.namedWindow("image_original", flags=cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("image_assembly", flags=cv2.WINDOW_AUTOSIZE)
        image_original = self.imageFromSlice(self.slices_part_image)
        tmp_slices = self.finishImageBuild()
        image_assembly = self.imageFromSlice(tmp_slices)
        while not None:
            cv2.imshow("image_original", image_original)
            cv2.imshow("image_assembly", image_assembly)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyWindow("image_original")
                cv2.destroyWindow("image_assembly`")
                break

    @staticmethod
    def showImage(image, tested=1):
        if not tested:
            return
        cv2.imshow("test image", image)
        cv2.waitKey(0)


def main(robot_ip, port=9559, count_parts=(4, 3)):
    controller = Controller(robot_ip, port, count_parts)
    tested = 0
    # controller.startEpisode(tested)
    # controller.testSay()
    # controller.testAnswer()
    # controller.testGetPhoto()
    # controller.testHeadMotion()
    # controller.testSensors()


def testIp(ip):
    tmp = 0
    dots = 0
    for i in ip:
        if i == ".":
            if tmp == 0:
                return 0
            else:
                tmp = 0
                dots += 1
            continue
        if str.isdigit(i):
            tmp += 1
            if tmp == 4:
                return 0
        else:
            return 0
    if dots != 3:
        return 0
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", type=str, default="127.0.0.1", help="Robot IP address.")
    parser.add_argument("-port", "-p", type=int, default=9559, help="Naoqi port number")
    args = vars(parser.parse_args())
    if not testIp(args["ip"]):
        print "bad format: 'ip'"
        print "program exit"
        sys.exit(1)
    # print(args)
    part_count = (4, 3)
    try:
        main(args["ip"], args["port"], part_count)
    except Exception as ex:
        print ex
        sys.exit(1)
