#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud=ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)

    # Set threshold scale factor
    x = .2
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling

    
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size   
    # Experiment and find the appropriate size!
    LEAF_SIZE = .005  

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered_vox = vox.filter()

    # TODO: PassThrough Filter
    
    passthrough = cloud_filtered_vox.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1
    passthrough.set_filter_limits(axis_min, axis_max)


    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered_pass = passthrough.filter()
    #filter y
    passthrough = cloud_filtered_pass.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = .35
    axis_max = 1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered_pass = passthrough.filter()
    

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered_pass.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = .012	
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    extracted_inliers = cloud_filtered_pass.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered_pass.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    cloud_objects_white=XYZRGB_to_XYZ(extracted_outliers)
    tree = cloud_objects_white.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = cloud_objects_white.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(.03)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(3000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([cloud_objects_white[indice][0],
                                            cloud_objects_white[indice][1],
                                            cloud_objects_white[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    #ros_cloud_filtered=pcl_to_ros(extracted_outliers)
    #ros_cloud_filtered2=pcl_to_ros(cloud_filtered_pass)
    ros_cloud_filtered3=pcl_to_ros(cluster_cloud)
	

    # TODO: Publish ROS messages
    #pcl_test_pub.publish(ros_cloud_filtered)
    #pcl_test2_pub.publish(ros_cloud_filtered2)
    pcl_test3_pub.publish(ros_cloud_filtered3)

# Exercise-3 TODOs:
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster_object = extracted_outliers.extract(pts_list)


        # TODO: convert the cluster from pcl to ROS using helper function
        cluster_ros=pcl_to_ros(pcl_cluster_object)


        # Extract histogram features
        chists = compute_color_histograms(cluster_ros,nbins=32 ,using_hsv=True)
        normals = get_normals(cluster_ros)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])
        # TODO: complete this step just as is covered in capture_features.py

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(cloud_objects_white[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = cluster_ros
        detected_objects.append(do)



    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    
    # TODO: Initialize variables
    #declare
    
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    #initialize
    test_scene_num.data = 2
    

    # TODO: Get/Read parameters

    
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')


    # TODO: Parse parameters into individual variables
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    dict_list = []
    centroid =[]

    

    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid=np.mean(points_arr, axis=0)[:3]
        centroid = [np.asscalar(centroid[0]),np.asscalar(centroid[1]),np.asscalar(centroid[2])]
        centroids.append(centroid)

    

    # TODO: Create 'place_pose' for the object
    
    for i in range(0, len(object_list_param)):
        object_name_l = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        for i in range(0,len(labels)):
                detected_object_name=labels[i]
                if(object_name_l==detected_object_name):
                    print('object deteted \n')
                    object_name.data=object_name_l
                    pick_pose.position.x=centroids[i][0]
                    pick_pose.position.y=centroids[i][1]
                    pick_pose.position.z=centroids[i][2]
                    for i in range(0,len(dropbox_list_param)):
                        dropbox_name=dropbox_list_param[i]['name']
                        dropbox_group=dropbox_list_param[i]['group']
                        dropbox_pos=dropbox_list_param[i]['position']
                        if(object_group==dropbox_group):
                            arm_name.data=dropbox_name
                            place_pose.position.x=dropbox_pos[0]
                            place_pose.position.y=dropbox_pos[1]
                            place_pose.position.z=dropbox_pos[2]
                    yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                    dict_list.append(yaml_dict)
    send_to_yaml('out2.yml', dict_list)
		
		
		# Wait for 'pick_place_routine' service to come up
                #rospy.wait_for_service('pick_place_routine')

		#try:
		    #pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

		    # TODO: Insert your message variables to be sent as a service request
		    #resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

		    #print ("Response: ",resp.success)

		#except rospy.ServiceException, e:
		    #print "Service call failed: %s"%e
	
    
      
    

# TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('object_detection', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size = 1)

    # TODO: Create Publishers
    #pcl_test_pub=rospy.Publisher("/outlier", PointCloud2, queue_size = 1)
    #pcl_test2_pub=rospy.Publisher("/inlier", PointCloud2, queue_size = 1)
    pcl_test3_pub=rospy.Publisher("/cluster", PointCloud2, queue_size = 1)

    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker,queue_size=1)



    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
