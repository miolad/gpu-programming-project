fn main() {
    cbindgen::Builder::new()
        .with_crate(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .with_language(cbindgen::Language::Cxx)
        .with_namespace("viewer")
        .generate()
        .unwrap()
        .write_to_file("cudaviewer.h");
}
