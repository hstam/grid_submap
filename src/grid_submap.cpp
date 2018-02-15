#include <chrono>
#include <sstream>
#include <fstream>
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
            cout << "Nice, now tell me which layers to save (comma separated):" << endl;
            cin>>input;

            stringstream ss(input);
            vector<string> v;
            string tmp_s;
            while(getline(ss, tmp_s, ',')){
                v.push_back(tmp_s);
            }

            long int now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            stringstream().swap(ss);
            ss << now;
            tmp_s = ss.str();
            string extension = "_submap.txt";

            ofstream ofs;
            unsigned szx = tmp_.getSize()[0];
            unsigned szy = tmp_.getSize()[1];
            for(const string& layer_name: v) {
                ofs.open(path + layer_name + "_" + tmp_s + extension, ios_base::app);
                ofs << szx << endl;
                ofs << szy << endl;
                unsigned cnt = 0;
                try{
                    for (grid_map::GridMapIterator iterator(tmp_); !iterator.isPastEnd(); ++iterator) {
                        if(cnt < szx){
                            if(cnt > 0){
                                ofs << "#";
                            }
                            cnt++;
                        }
                        else{
                            cnt = 1;
                            ofs << endl;
                        }
                        ofs << tmp_.at(layer_name, *iterator);
                    }
                }
                catch(...){
                    cout << "\033[1;31mCould not fine layer name: " << layer_name <<"!\033[0m" << endl;
                }
                ofs.close();
                std::cout << "\033[1;32mWrote\033[0m \033[1;31m" + path + layer_name + "_" + tmp_s + extension + "\033[0m \033[1;32mto disk!\033[0m" << std::endl;
            }


            break;
        }
    }
    more = 1;
}


int main (int argc, char** argv){
    ros::init (argc, argv, "grid_submap");
    ros::NodeHandle nh;

    string in_topic;
    nh.param("grid_submap/input_topic", in_topic, string("/traversability_estimation/traversability_map"));

    path = ros::package::getPath("grid_submap") + "/data/";

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