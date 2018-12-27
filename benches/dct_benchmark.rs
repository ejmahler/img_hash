#![feature(test)]
extern crate test;
extern crate img_hash;
extern crate rustdct;
use rustdct::algorithm::type2and3_butterflies::Type2And3Butterfly8;

fn bench_naive(b: &mut test::Bencher, width: usize, height: usize) {

    let mut signal = vec![0f64; width * height];

    b.iter(|| { img_hash::dct_2d(&mut signal, width); });
}
#[bench] fn naive_impl_004x004(b: &mut test::Bencher) { bench_naive(b, 4, 4); }
#[bench] fn naive_impl_008x008(b: &mut test::Bencher) { bench_naive(b, 8, 8); }
#[bench] fn naive_impl_016x016(b: &mut test::Bencher) { bench_naive(b, 16, 16); }
//#[bench] fn naive_impl_256x256(b: &mut test::Bencher) { bench_naive(b, 256, 256); }




const BLOCK_SIZE: usize = 8;

#[inline(always)]
unsafe fn transpose_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, block_x: usize, block_y: usize) {
    for inner_x in 0..BLOCK_SIZE {
        for inner_y in 0..BLOCK_SIZE {
            let x = block_x * BLOCK_SIZE + inner_x;
            let y = block_y * BLOCK_SIZE + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

#[inline(always)]
unsafe fn transpose_endcap_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, block_x: usize, block_y: usize, block_width: usize, block_height: usize) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = block_x * BLOCK_SIZE + inner_x;
            let y = block_y * BLOCK_SIZE + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
// Use "Loop tiling" to improve cache-friendliness
pub fn transpose<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
	assert_eq!(width*height, input.len());
	assert_eq!(width*height, output.len());

    let x_block_count = width / BLOCK_SIZE;
    let y_block_count = height / BLOCK_SIZE;

    let remainder_x = width - x_block_count * BLOCK_SIZE;
    let remainder_y = height - y_block_count * BLOCK_SIZE;

    for y_block in 0..y_block_count {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input, output,
                    width, height, 
                    x_block, y_block);
            }
        }

        //if the width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_endcap_block(
                    input, output, 
                    width, height, 
                    x_block_count, y_block, 
                    remainder_x, BLOCK_SIZE);
            }
        }
    }

    //if the height is not cleanly divisible by BLOCK_SIZE, there are still a few columns that haven't been transposed
    if remainder_y > 0 {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height,
                    x_block, y_block_count,
                    BLOCK_SIZE, remainder_y,
                    );
            }
        }

        //if the width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_endcap_block(
                    input, output,
                    width, height, 
                    x_block_count, y_block_count, 
                    remainder_x, remainder_y);
            }
        }
    } 
}


fn external_dct(planner: &mut rustdct::DCTplanner<f64>, data: &mut [f64], width: usize, height: usize) {
	let width_dct = planner.plan_dct2(width);
	let height_dct = planner.plan_dct2(height);

	let mut scratch = vec![0f64; data.len()];

	// width DCT
	for (src, dest) in data.chunks_exact_mut(width).zip(scratch.chunks_exact_mut(width)) {
		width_dct.process_dct2(src, dest);
	}

	// transpose
	transpose(width, height, &scratch, data);

	// height DCT
	for (src, dest) in data.chunks_exact_mut(height).zip(scratch.chunks_exact_mut(height)) {
		height_dct.process_dct2(src, dest);
	}

	// transpose back
	transpose(height, width, &scratch, data);
}

fn bench_external(b: &mut test::Bencher, width: usize, height: usize) {
	let mut planner = rustdct::DCTplanner::new();
    let mut signal = vec![0f64; width * height];

    b.iter(|| { external_dct(&mut planner, &mut signal, width, height); });
}

#[bench] fn external_impl_004x004(b: &mut test::Bencher) { bench_external(b, 4, 4); }
#[bench] fn external_impl_008x008(b: &mut test::Bencher) { bench_external(b, 8, 8); }
#[bench] fn external_impl_016x016(b: &mut test::Bencher) { bench_external(b, 16, 16); }
#[bench] fn external_impl_256x256(b: &mut test::Bencher) { bench_external(b, 256, 256); }



fn external_dct_no_transpose(planner: &mut rustdct::DCTplanner<f64>, data: &mut [f64], width: usize, height: usize) {
	let width_dct = planner.plan_dct2(width);
	let height_dct = planner.plan_dct2(height);

	let mut scratch = vec![0f64; data.len()];

	// width DCT
	for (src, dest) in data.chunks_exact_mut(width).zip(scratch.chunks_exact_mut(width)) {
		width_dct.process_dct2(src, dest);
	}

	// transpose
	// transpose(width, height, &scratch, data);

	// height DCT
	for (src, dest) in data.chunks_exact_mut(height).zip(scratch.chunks_exact_mut(height)) {
		height_dct.process_dct2(src, dest);
	}

	// transpose back
	// transpose(height, width, &scratch, data);
}

fn bench_external_no_transpose(b: &mut test::Bencher, width: usize, height: usize) {
	let mut planner = rustdct::DCTplanner::new();
    let mut signal = vec![0f64; width * height];

    b.iter(|| { external_dct_no_transpose(&mut planner, &mut signal, width, height); });
}

#[bench] fn external_no_transpose_impl_004x004(b: &mut test::Bencher) { bench_external_no_transpose(b, 4, 4); }
#[bench] fn external_no_transpose_impl_008x008(b: &mut test::Bencher) { bench_external_no_transpose(b, 8, 8); }
#[bench] fn external_no_transpose_impl_016x016(b: &mut test::Bencher) { bench_external_no_transpose(b, 16, 16); }
#[bench] fn external_no_transpose_impl_256x256(b: &mut test::Bencher) { bench_external_no_transpose(b, 256, 256); }

fn fixed_dct8x8(dct_instance: &Type2And3Butterfly8<f64>, data: &mut [f64]) {
	let mut scratch = vec![0f64; 64];

	// width DCT
	for row in data.chunks_exact_mut(8) {
		unsafe { dct_instance.process_inplace_dct2(row) };
	}

	// transpose
	transpose(8, 8, data, &mut scratch);

	// height DCT
	for col in scratch.chunks_exact_mut(8) {
		unsafe { dct_instance.process_inplace_dct2(col) };
	}

	// transpose back
	transpose(8, 8, &scratch, data);
}

#[bench]
fn bench_fixed_8(b: &mut test::Bencher) {
	let dct_instance = Type2And3Butterfly8::new();
    let mut signal = vec![0f64; 8 * 8];

    b.iter(|| { fixed_dct8x8(&dct_instance, &mut signal); });
}
