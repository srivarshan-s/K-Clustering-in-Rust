// TODO: Implement mode clustering
// TODO: Implement prediction function
// TODO: Implement elbow curve method to find the optimal number of clusters

use rand::Rng;

#[derive(Debug)]
enum ClusterMethod {
    Mean,
    Mode,
}

#[derive(Debug)]
pub struct KClustering {
    method: ClusterMethod,
    cluster_points: Vec<Vec<f32>>,
}

impl KClustering {
    pub fn new(method: &str) -> KClustering {
        let kcluster: KClustering = match method {
            "mean" => KClustering {
                method: ClusterMethod::Mean,
                cluster_points: Vec::<Vec<f32>>::new(),
            },
            "mode" => {
                println!("MODE method not implemented yet");
                println!("Defaulting to MEAN method");
                KClustering {
                    method: ClusterMethod::Mode,
                    cluster_points: Vec::<Vec<f32>>::new(),
                }
            }
            _ => {
                println!("Wrong method arguements");
                println!("Defaulting to MEAN method");
                KClustering {
                    method: ClusterMethod::Mean,
                    cluster_points: Vec::<Vec<f32>>::new(),
                }
            }
        };
        kcluster
    }

    pub fn fit(&mut self, num_clusters: &i32, feat: &[Vec<f32>], max_iter: i32) {
        self.init_cluster_points(num_clusters, feat);
        self.cluster(feat, max_iter);
    }

    pub fn predict(&self, test_feat: &Vec<Vec<f32>>) {
        println!("Prediction function");
    }

    fn init_cluster_points(&mut self, num_clusters: &i32, feat: &[Vec<f32>]) {
        let mut rng = rand::thread_rng();
        let mut rand_nums = Vec::<i32>::new();
        for _i in 0..*num_clusters {
            let mut num = rng.gen_range(0..feat.len());
            while rand_nums.contains(&(num as i32)) {
                num = rng.gen_range(0..feat.len());
            }
            rand_nums.push(num as i32);
        }
        for i in 0..*num_clusters {
            let index = rand_nums[i as usize] as usize;
            let temp = &feat[index];
            self.cluster_points.push(temp.to_vec());
        }
    }

    fn cluster(&mut self, feat: &[Vec<f32>], iter_num: i32) {
        // dbg!(&iter_num);
        if iter_num == 0 {
            return;
        }
        let mut cluster_new = Vec::<Vec<Vec<f32>>>::new();
        for _i in 0..self.cluster_points.len() {
            cluster_new.push(vec![vec![0.0]]);
        }
        for feat_item in feat.iter() {
            let mut min_dist: f32 = self.get_dist(&self.cluster_points[0], feat_item);
            let mut min_dist_idx = 0;
            for j in 0..self.cluster_points.len() {
                if self.get_dist(&self.cluster_points[j], feat_item) < min_dist {
                    min_dist = self.get_dist(&self.cluster_points[j], feat_item);
                    min_dist_idx = j;
                }
            }
            cluster_new[min_dist_idx].push(feat_item.clone());
        }
        for cluster_item in cluster_new.iter_mut() {
            cluster_item.remove(0);
        }
        let mut cluster_points_new = Vec::<Vec<f32>>::new();
        for _i in 0..self.cluster_points.len() {
            cluster_points_new.push(vec![0.0, 0.0]);
        }
        match self.method {
            ClusterMethod::Mean => {
                cluster_points_new = self.mean(cluster_new, &mut cluster_points_new)
            }
            ClusterMethod::Mode => {
                cluster_points_new = self.mode(cluster_new, &mut cluster_points_new)
            }
        }
        if self.cluster_check(&self.cluster_points, &cluster_points_new) {
            return;
        }
        self.cluster_points = cluster_points_new;
        self.cluster(feat, iter_num-1)
    }

    fn cluster_check(&self, cluster_points_1: &Vec<Vec<f32>>, cluster_points_2: &Vec<Vec<f32>>) -> bool {
        let mut flag: i32 = 0;
        for i in 0..cluster_points_1.len() {
            if cluster_points_1[i] != cluster_points_2[i] {
                flag = 1;
            }
        }
        if flag == 1 {
            false
        } else {
            true
        }
    }

    fn mean(
        &self,
        cluster_new: Vec<Vec<Vec<f32>>>,
        cluster_points_new: &mut Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        for i in 0..cluster_points_new.len() {
            for j in 0..cluster_points_new[i].len() {
                let mut sum: f32 = 0 as f32;
                for k in 0..cluster_new[i].len() {
                    sum += cluster_new[i][k][j];
                }
                sum /= cluster_new[i].len() as f32;
                cluster_points_new[i][j] = sum;
            }
        }
        cluster_points_new.to_vec()
    }

    // Mode function same as mean for now
    fn mode(
        &self,
        cluster_new: Vec<Vec<Vec<f32>>>,
        cluster_points_new: &mut Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        for i in 0..cluster_points_new.len() {
            for j in 0..cluster_points_new[i].len() {
                let mut sum: f32 = 0 as f32;
                for k in 0..cluster_new[i].len() {
                    sum += cluster_new[i][k][j];
                }
                sum /= cluster_new[i].len() as f32;
                cluster_points_new[i][j] = sum;
            }
        }
        cluster_points_new.to_vec()
    }
    
    fn get_dist(&self, point1: &[f32], point2: &[f32]) -> f32 {
        let mut sum: f32 = 0 as f32;
        for i in 0..point1.len() {
            sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        sum = sum.sqrt();
        sum
    }
}