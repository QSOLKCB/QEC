use std::{env, fs, path::PathBuf};

fn main() {
    // QEC's project version is the release authority; avoid a second manual bump.
    println!("cargo:rerun-if-changed=../pyproject.toml");
    let path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../pyproject.toml");
    let text = fs::read_to_string(path).expect("cannot read QEC pyproject.toml");
    let mut in_project = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_project = line == "[project]";
        } else if in_project {
            if let Some((key, value)) = line.split_once('=') {
                if key.trim() == "version" {
                    let version = value.trim().trim_matches('"');
                    let parts: Vec<_> = version.split('.').collect();
                    assert!(parts.len() == 3 && parts.iter().all(|part| {
                        !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit())
                    }), "QEC project version must be numeric major.minor.patch");
                    println!("cargo:rustc-env=QEC_RELEASE_VERSION={version}");
                    return;
                }
            }
        }
    }
    panic!("QEC pyproject.toml is missing [project].version");
}
