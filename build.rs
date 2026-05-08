fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_TARPAULIN").is_some()
        || std::env::var_os("CARGO_TARPAULIN").is_some()
        || std::env::var_os("CARGO_CFG_TARPAULIN").is_some()
    {
        println!("cargo:rustc-cfg=tarpaulin");
    }
}
