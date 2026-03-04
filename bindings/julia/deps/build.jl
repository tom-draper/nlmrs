# Build script for nlmrs.jl — compiles the nlmrs-c Rust crate from source.
# Runs automatically on `Pkg.add` / `Pkg.build`.
# Requires Rust and Cargo to be installed: https://rustup.rs

c_dir      = abspath(joinpath(@__DIR__, "..", "..", "c"))
cargo_toml = joinpath(c_dir, "Cargo.toml")

isfile(cargo_toml) ||
    error("Cannot find C bindings at $c_dir. " *
          "Ensure the full nlmrs repository is available.")

try
    run(`cargo build --release --manifest-path $cargo_toml`)
catch e
    error("Cargo build failed — is Rust installed? (https://rustup.rs)\n$e")
end

libname = if Sys.iswindows()
    "nlmrs_c.dll"
elseif Sys.isapple()
    "libnlmrs_c.dylib"
else
    "libnlmrs_c.so"
end

libpath = abspath(joinpath(c_dir, "target", "release", libname))

isfile(libpath) ||
    error("Build succeeded but library not found at $libpath")

open(joinpath(@__DIR__, "deps.jl"), "w") do f
    println(f, "const _libpath = $(repr(libpath))")
end

println("nlmrs: built $libpath")
