#include <chrono>
#include <sstream>
#include <algorithm>
#include <ros/ros.h>
#include <ros/package.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include "grid_map_core/SubmapGeometry.hpp"

using namespace std;
using namespace grid_map;

ros::Publisher pub;
string in_topic, path;
int more;

void gridMapCallback (const grid_map_msgs::GridMap& msg){
    more = 0;

    GridMap map_;
    GridMap tmp_;

    GridMapRosConverter::fromMessage(msg, map_);
    GridMapRosConverter::fromMessage(msg, tmp_);

    while(true){
        cout << "\033[1;33m====================================\033[0m" << endl;
        cout << "\033[1;33m===========\033[0m \033[1;31mGrid Submap!\033[0m \033[1;33m===========\033[0m" << endl;
        cout << "\033[1;33m===============\033[0m \033[1;31mHelp\033[0m \033[1;33m===============\033[0m" << endl;
        cout << "\033[1;33m====================================\033[0m" << endl;
        cout << "\033[1;34m-q\033[0m \033[1;34m--quit\033[0m: quit program" << endl;
        string input;
        double x;
        while(input != "-q" and input != "--quit"){
            cout << "- Select the \033[1;33mcenter point on the X\033[0m axis: -\n";
            cin>>input;
            try{
                x = stod(input);
                break;
            }
            catch(...){
                continue;
            }
        }
        if(input == "-q" or input == "--quit"){
            more = -1;
            return;
        }
        double y;
        while(input != "-q" and input != "--quit"){
            cout << "- Select the \033[1;33mcenter point on the Y\033[0m axis: -\n";
            cin>>input;
            try{
                y = stod(input);
                break;
            }
            catch(...){
                continue;
            }
        }
        if(input == "-q" or input == "--quit"){
            more = -1;
            return;
        }
        Position p(x, y);

        double lx;
        while(input != "-q" and input != "--quit"){
            cout << "- Select the \033[1;33mlength in meters on the X\033[0m axis: -\n";
            cin>>input;
            try{
                lx = stod(input);
                break;
            }
            catch(...){
                continue;
            }
        }
        if(input == "-q" or input == "--quit"){
            more = -1;
            return;
        }
        double ly = 0.0;
        while(input != "-q" and input != "--quit"){
            cout << "- Select the \033[1;33mlength in meters on the Y\033[0m axis: -\n";
            cin>>input;
            try{
                ly = stod(input);
                break;
            }
            catch(...){
                continue;
            }
        }
        if(input == "-q" or input == "--quit"){
            more = -1;
            return;
        }

        Length l(lx, ly);

        bool success;
        tmp_ = map_.getSubmap(p, l, success);
        grid_map_msgs::GridMap msg_;
        GridMapRosConverter::toMessage(tmp_, msg_);
        pub.publish(msg_);
        while(input != "yes" and input != "y" and input!="no" and input!= "n"){
            cout << "Are you satisfied with the grid submap? ((y)es/(n)o)" << endl;
            cin>>input;
            transform(input.begin(), input.end(), input.begin(), ::tolower);
        }
        if(input == "yes" or input == "y"){
            // TODO Write to file!
            cout << "Placeholder for saved file message" << endl;
            break;
        }
    }
    more = 1;
    //map_.getSubmap()
    /*
    pcl::PCLPointCloud2 pcl_cloud;
    pcl_conversions::toPCL(msg, pcl_cloud);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromPCLPointCloud2(pcl_cloud, cloud);
    pcl::PointCloud<pcl::PointXYZRGB> cloud2 = pcl::PointCloud<pcl::PointXYZRGB>(cloud);

    for(unsigned i=0; i<cloud.width; i+=sy){
        for(unsigned j=0; j<cloud.height; j+=sx){
            unsigned no_nan = 0;
            double maxw = 0;
            double maxl = 0;
            unsigned maxi = 0;
            unsigned maxj = 0;
            for(unsigned l=j;l<cloud.height;l++){
                for(unsigned k=i;k<cloud.width;k++){
                    if(!(isnan(cloud.at(k,l).x) or 
                        isnan(cloud.at(k,l).y) or 
                        isnan(cloud.at(k,l).z) or 
                        isinf(cloud.at(k,l).x) or 
                        isinf(cloud.at(k,l).y) or 
                        isinf(cloud.at(k,l).z) or 
                        isnan(cloud.at(i,j).x) or 
                        isnan(cloud.at(i,j).y) or 
                        isnan(cloud.at(i,j).z) or 
                        isinf(cloud.at(i,j).x) or 
                        isinf(cloud.at(i,j).y) or 
                        isinf(cloud.at(i,j).z))){
                        no_nan++;
                        double ll = cloud.at(i,j).x - cloud.at(k,l).x;
                        double w = cloud.at(i,j).y - cloud.at(k,l).y;
                        // I can't get abs to work?! wtf!
                        if(ll < 0){
                            ll = -ll;
                        }
                        if(w < 0){
                            w = -w;
                        }
                        if(k > maxi){
                            maxi = k;
                        }
                        if(l > maxj){
                            maxj = l;
                        }
                        if(ll <= length and w <= width){
                            cloud.at(k,l).r = red_value;
                            cloud.at(k,l).g = green_value;
                            cloud.at(k,l).b = blue_value;
                            if(ll > maxl){
                                maxl = ll;
                            }
                            if(w > maxw){
                                maxw = w;
                            }
                        }
                        if(abs(cloud.at(i,j).x - cloud.at(k,l).x) > length and abs(cloud.at(i,j).y - cloud.at(k,l).y) > width){
                            k = cloud.width;
                            l = cloud.height;
                        }
                    }
                }
            }
            if(maxl >= length-tolerance and maxw >= width-tolerance){
                sensor_msgs::PointCloud2 output;
                pcl::toPCLPointCloud2(cloud, pcl_cloud);
                pcl_conversions::fromPCL(pcl_cloud, output);
                pub.publish (output);
                cloud = pcl::PointCloud<pcl::PointXYZRGB>(cloud2);
                cout << "\033[1;33m====================================\033[0m" << endl;
                cout << "\033[1;33m=======\033[0m \033[1;31mPointCloud Annotator\033[0m \033[1;33m=======\033[0m" << endl;
                cout << "\033[1;33m===============\033[0m \033[1;31mHelp\033[0m \033[1;33m===============\033[0m" << endl;
                cout << "\033[1;33m====================================\033[0m" << endl;
                cout << "\033[1;34m-q\033[0m \033[1;34m--quit\033[0m: quit program" << endl;
                cout << "\033[1;34m-s\033[0m \033[1;34m--skip\033[0m: skip current points" << endl;
                cout << "\033[1;34mAnything else\033[0m: annotation name" << endl;
                cout << "- Annotate the \033[1;33mhighlighted\033[0m points: -\n";
                string input;
                cin>>input;
                transform(input.begin(), input.end(), input.begin(), ::tolower);
                if(input == "--quit" or input == "-q"){
                    i = cloud.width;
                    j = cloud.height;
                    more = -1;
                }
                else if(input != "--skip" and input != "-s"){
                    pcl::PointCloud<pcl::PointXYZRGB> tmpcloud;
                    for(unsigned jj=j; jj<maxj; jj++){
                        for(unsigned ii=i; ii<maxi; ii++){
                            tmpcloud.push_back(cloud.at(ii,jj));
                        }
                    }
                    tmpcloud.width = maxi - i;
                    tmpcloud.height = maxj - j;
                    long int now = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
                    string s;
                    stringstream ss;
                    ss << now;
                    s = ss.str();
                    pcl::io::savePCDFileASCII (path+"/data/"+input+"_"+s+".pcd", tmpcloud);
                    cout << "\033[1;32mWrote\033[0m \033[1;31m" + path+"/data/"+input+"_"+s+".pcd" + "\033[0m \033[1;32mto disk!\033[0m" << endl;
                }
            }
        }
    }
    if(more == 0){
        string input = "answer";
        while(input != "yes" and input != "y" and input!="no" and input!= "n"){
            cout << "This pointcloud has been annotated. Do you want to continue to the next one? ((y)es/(n)o)" << endl;
            cin>>input;
            transform(input.begin(), input.end(), input.begin(), ::tolower);
        }
        if(input == "yes" or input == "y"){
            more = 1;
        }
        else{
            more = -1;
        }
    }
    */
}


int main (int argc, char** argv){
    ros::init (argc, argv, "grid_submap");
    ros::NodeHandle nh;

    string in_topic;
    nh.param("grid_submap/input_topic", in_topic, string("/elevation_mapping/elevation_map"));

    path = ros::package::getPath("grid_submap");

    ros::Subscriber sub = nh.subscribe (in_topic, 1, gridMapCallback);

    pub = nh.advertise<grid_map_msgs::GridMap> ("grid_submap/requested_submap", 1);

    more = true;
    while(ros::ok()){
        if(more == 1){
            ros::spinOnce();
        }
        else if(more == -1){
            cout << "Goodbye!" << endl;
            break;
        }
    }
}