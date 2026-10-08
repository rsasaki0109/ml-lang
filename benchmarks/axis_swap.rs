//! Isolated kernel ablation: historical element-allocation permutation vs suffix blocks.
#[allow(dead_code)]
#[path = "../src/tensor.rs"]
mod tensor;
use std::{hint::black_box, time::Instant};
use tensor::{Op, Tensor};
fn legacy_index(mut flat: usize, shape: &[usize], first: usize, second: usize) -> usize {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    coordinates.swap(first, second);
    let mut index = 0;
    for (axis, coord) in coordinates.iter().copied().enumerate() {
        let dimension = if axis == first {
            shape[second]
        } else if axis == second {
            shape[first]
        } else {
            shape[axis]
        };
        index = index * dimension + coord;
    }
    index
}
fn execute(
    legacy: bool,
    backward: bool,
    x: &Tensor,
    y: &Tensor,
    g: &[f64],
    first: usize,
    second: usize,
) -> Vec<f64> {
    if !legacy {
        if backward {
            return tensor::vjp(Op::SwapAxes { first, second }, x, None, y, g).0;
        }
        return tensor::forward(Op::SwapAxes { first, second }, x, None)
            .unwrap()
            .data;
    }
    let mut data = vec![0.; x.data.len()];
    if backward {
        for (i, &v) in g.iter().enumerate() {
            data[legacy_index(i, &y.shape, first, second)] = v;
        }
    } else {
        for (i, dst) in data.iter_mut().enumerate() {
            *dst = x.data[legacy_index(i, &y.shape, first, second)];
        }
        return Tensor::new(y.shape.clone(), data).unwrap().data;
    }
    data
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4);
    let legacy = match args[1].as_str() {
        "legacy" => true,
        "blocks" => false,
        _ => panic!("backend"),
    };
    let backward = match args[2].as_str() {
        "forward" => false,
        "vjp" => true,
        _ => panic!("mode"),
    };
    let (shape, first, second) = match args[3].as_str() {
        "heads" => (vec![8, 64, 8, 32], 1, 2),
        "last" => (vec![8, 8, 64, 32], 2, 3),
        "identity" => (vec![8, 8, 64, 32], 1, 1),
        _ => panic!("workload"),
    };
    let x = Tensor::new(
        shape,
        (0..131072).map(|i| (i % 997) as f64 / 17. - 4.).collect(),
    )
    .unwrap();
    let mut outshape = x.shape.clone();
    outshape.swap(first, second);
    let y = Tensor::new(outshape, vec![0.; x.data.len()]).unwrap();
    let g: Vec<_> = (0..x.data.len())
        .map(|i| (i % 991) as f64 / 19. - 3.)
        .collect();
    black_box(execute(legacy, backward, &x, &y, &g, first, second));
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(execute(legacy, backward, &x, &y, &g, first, second));
    }
    let ms = start.elapsed().as_secs_f64() * 1000. / iterations as f64;
    let result = execute(legacy, backward, &x, &y, &g, first, second);
    for i in 0..x.data.len() {
        let input = legacy_index(i, &y.shape, first, second);
        if backward {
            assert_eq!(result[input], g[i]);
        } else {
            assert_eq!(result[i], x.data[input]);
        }
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let rss = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<usize>()
        .unwrap();
    println!("{{\"milliseconds\":{ms},\"peak_rss_kib\":{rss},\"iterations\":{iterations},\"elements\":{}}}",x.data.len());
}
