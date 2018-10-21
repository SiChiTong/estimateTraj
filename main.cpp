#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

void loadImgNames ( const std::string& img_folder, std::vector<std::string>& img_names )
{
    std::ifstream file;
    std::string img_name_file = img_folder + "/img_names.txt";
    file.open ( img_name_file.c_str() );
    while ( !file.eof() ) {
        std::string s;
        std::getline ( file, s );
        if ( !s.empty() ) {
            img_names.push_back (s);
        }// 如果有数据
    }// 读取整个文件
}

int main ( int argc, char **argv )
{

    if ( argc != 4 ) {
        std::cout << "Please input: cfg_dir and imgs_dir and out_dir\n\n";
        return -1;
    }

    // local parameters
    std::string cfg_dir = argv[1];
    cv::FileStorage fs ( cfg_dir, cv::FileStorage::READ );

    // camera params
    float fx = fs["camera.fx"];
    float fy = fs["camera.fy"];
    float cx = fs["camera.cx"];
    float cy = fs["camera.cy"];

    float k1 = fs["camera.k1"];
    float k2 = fs["camera.k2"];
    float p1 = fs["camera.p1"];
    float p2 = fs["camera.p2"];
    float k3 = fs["camera.k3"];

    cv::Mat K = ( cv::Mat_<float> ( 3, 3 ) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 );
    cv::Mat dist = ( cv::Mat_<float> ( 1, 5 ) << k1, k2, p1, p2, k3 );
    Eigen::Matrix3d eK;
    eK << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    // aruco marker params
    int n_markers = fs["aruco.n_markers"];
    int marker_size = fs["aruco.marker_size"];
    double marker_length = fs["aruco.marker_length"];

    // Homograph matrix
    cv::Mat cvH ;
    fs["homograph_matrix"] >> cvH;
    Eigen::Matrix3d H;
    H << cvH.at<double> ( 0, 0 ), cvH.at<double> ( 0, 1 ), cvH.at<double> ( 0, 2 ),
      cvH.at<double> ( 1, 0 ), cvH.at<double> ( 1, 1 ), cvH.at<double> ( 1, 2 ),
      cvH.at<double> ( 2, 0 ), cvH.at<double> ( 2, 1 ), cvH.at<double> ( 2, 2 );
    Eigen::Matrix3d H_inv = H.inverse();

    // img_dirs
    std::string img_folder = argv[2];

    // output dir
    std::string traj_dir = argv[3];

    // out_traj
    std::ofstream   traj_file;
    traj_file.open ( traj_dir );

    // init aruco dictionary
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::generateCustomDictionary ( n_markers, marker_size );

    // read all images
    std::vector<std::string> img_names;
    loadImgNames ( img_folder, img_names );
    for ( size_t i = 0; i < img_names.size(); i ++ ) {
        // read img
		
        std::string& sname = img_names.at ( i );
		std::string name = img_folder + "/"+ sname + ".bmp";
		std::string timestamp = sname;
        cv::Mat img = cv::imread ( name, -1 );
        cv::cvtColor ( img, img, cv::COLOR_GRAY2BGR );

        // detect aruco marker
        std::vector< std::vector<cv::Point2f> > marker_corners;
        std::vector<int> IDs;
        cv::aruco::detectMarkers ( img, dictionary , marker_corners, IDs );
        cv::aruco::drawDetectedMarkers ( img, marker_corners, IDs );
        if ( IDs.size() != 1 ) {
            continue;
        }

        // get the four corners
        std::vector<Eigen::Vector2d> pts;
        std::vector< cv::Point2f >& corners = marker_corners.at ( 0 );
        
        // undist
        std::vector<cv::Point2f> undist_corners;
        cv::undistortPoints ( corners, undist_corners, K, dist, cv::Mat(), K );
        
        
        for ( size_t np = 0; np < undist_corners.size(); np ++ ) {
            cv::Point2f& corner = undist_corners.at ( np );
            Eigen::Vector3d e_corner ( corner.x, corner.y, 1.0 );
            Eigen::Vector3d pt = H_inv * e_corner;
            pt = pt / pt[2];
            pts.push_back ( Eigen::Vector2d ( pt[0], pt[1] ) );
        }

        // check for corners
        double d1 = ( pts[0] - pts[1] ).norm();
        double d2 = ( pts[1] - pts[2] ).norm();
        double d3 = ( pts[2] - pts[3] ).norm();
        double d4 = ( pts[3] - pts[0] ).norm();
        double d_mean = ( d1 + d2 + d3 + d4 ) *0.25;
        if ( fabs ( ( d_mean - marker_length ) / marker_length ) > 0.2 ) {
            std::cout << "Bad length" << std::endl;
            continue;
        }

        // calc theta
        Eigen::Vector2d pta = pts[0] - pts[3];
        Eigen::Vector2d ptb = pts[1] - pts[2];
        double theta1 = atan2 ( pta[1], pta[0] );
        double theta2 = atan2 ( ptb[1], ptb[0] );
        double d_theta = theta1 -theta2;
        double theta = 0.0;
        if ( fabs ( d_theta ) > M_PI ) {
            if ( d_theta >  M_PI ) {
                d_theta -= 2.0 * M_PI;
            }
            if ( d_theta < -M_PI ) {
                d_theta += 2.0 * M_PI;
            }
            if ( fabs ( d_theta ) < 0.03 ) {
                theta = M_PI;
            } else {
                std::cout << "Bad theta\n";
                continue;
            }
        } else if ( fabs ( d_theta < 0.03 ) ) {
            theta = 0.5 * ( theta1 + theta2 );
        } else {
            std::cout << "Bad theta\n";
        }

        Eigen::Vector2d xy = ( pts[0] + pts[1] + pts[2] + pts[3] ) * 0.25;
		std::cout << timestamp << " " << xy[0] << " " << xy[1] << " " << theta << "\n";
// TODO second method
//         cv::Point2f cv_uv = 0.25 * ( corners[0] +  corners[1] + corners[2] + corners[3] );
//         Eigen::Vector3d e_uv1 ( cv_uv.x, cv_uv.y, 1.0 );
//         Eigen::Vector3d ept3 = H_inv * e_uv1;
//         ept3 = ept3 / ept3[2];
//         Eigen::Vector2d ptxy2 ( ept3[0], ept3[1] );
//
// 		if( ( ptxy1 - ptxy2 ).norm() > 0.002 )
// 			std::cout << "Bad two method\n";

		// save as TUM format
        Eigen::Matrix3d R;
        R << cos ( theta ), -sin ( theta ), 0.0,
          sin ( theta ), cos ( theta ), 0.0,
          0.0, 0.0, 1.0;
        Eigen::Quaterniond q ( R );
        std::vector<float> v ( 4 );
        v[0] = q.x();
        v[1] = q.y();
        v[2] = q.z();
        v[3] = q.w();
		traj_file << timestamp << " " <<  std::setprecision ( 9 ) << xy[0] << " " << xy[1] << " " << 0.0 << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		
        // show image
//         cv::Mat img_show;
//         cv::resize ( img, img_show, cv::Size ( img.cols / 2,  img.rows/2 ) );
//         cv::imshow ( "img_show", img_show );
//         cv::waitKey ( 1 );
    }

    return 0;
}
