use k_clustering::KClustering;

fn main() {
    let mut kmean = {
        println!("Initializing the Model");
        KClustering::new("mean")
    };
    let (num_clusters, feat) = hardcode_init();
    let max_iter: i32 = 100;
    println!("Fitting the Model");
    kmean.fit(&num_clusters, &feat, max_iter);
    dbg!(&kmean);
    let test_feat: Vec<Vec<f32>> = hardcode_test();
    let test_pred = kmean.predict(&test_feat);
    dbg!(test_pred);
}

fn hardcode_init() -> (i32, Vec<Vec<f32>>) {
    let num_clusters = 2;
    let feat = vec![
        vec![1.7, 1.8],
        vec![2.6, 1.3],
        vec![4.4, 3.7],
        vec![5.8, 4.1],
        vec![3.6, 2.8],
        vec![2.2, 5.7],
        vec![1.5, 3.3],
        vec![5.1, 1.9],
        vec![4.9, 2.7],
        vec![1.3, 4.5],
    ];
    (num_clusters, feat)
}

fn hardcode_test() -> Vec<Vec<f32>> {
    vec![
        vec![1.7, 1.8],
        vec![2.6, 1.3],
        vec![4.4, 3.7],
        vec![5.8, 4.1],
        vec![3.6, 2.8],
        vec![2.2, 5.7],
        vec![1.5, 3.3],
        vec![5.1, 1.9],
        vec![4.9, 2.7],
        vec![1.3, 4.5],
    ]
}
