#![no_std]
#![doc = include_str!("../README.md")]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/RustCrypto/media/6ee8e381/logo.svg",
    html_favicon_url = "https://raw.githubusercontent.com/RustCrypto/media/6ee8e381/logo.svg"
)]

#[cfg(not(miri))]
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub mod aarch64;

#[cfg(not(miri))]
#[cfg(target_arch = "loongarch64")]
#[doc(hidden)]
pub mod loongarch64;

#[cfg(not(miri))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(miri)]
mod miri;

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "loongarch64",
    target_arch = "x86",
    target_arch = "x86_64"
)))]
compile_error!("This crate works only on `aarch64`, `loongarch64`, `x86`, and `x86-64` targets.");

/// Create module with CPU feature detection code.
///
/// # Single target feature set
///
/// The module gets a `get` function returning whether all listed target features are
/// available:
///
/// ```
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// # fn main() {
/// cpufeatures::new!(aes_sha, "aes", "sha");
///
/// if aes_sha::get() {
///     // ...
/// }
/// # }
/// # #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
/// # fn main() {}
/// ```
///
/// # Multiple target feature sets
///
/// Several named sets can be declared instead, separated with `;` and followed by a
/// `_ => <variant>` entry naming the variant used when none of them is available. The module
/// then gets a `Features` enum with one variant per entry and `get` returns the first variant
/// whose target features are all available.
///
/// Detection is performed once for all sets and cached in a single atomic variable, so
/// dispatch costs one relaxed load instead of one per set.
///
/// ```
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// # fn main() {
/// cpufeatures::new!(
///     backend;
///     Avx2: "avx2", "aes";
///     Aes: "aes", "sse4.1";
///     _ => Soft;
/// );
///
/// use backend::Features;
///
/// match backend::get() {
///     Features::Avx2 => { /* ... */ }
///     Features::Aes => { /* ... */ }
///     Features::Soft => { /* ... */ }
/// }
/// # }
/// # #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
/// # fn main() {}
/// ```
#[macro_export]
macro_rules! new {
    ($mod_name:ident, $($tf:tt),+ $(,)?) => {
        mod $mod_name {
            use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

            const UNINIT: u8 = u8::MAX;
            static STORAGE: AtomicU8 = AtomicU8::new(UNINIT);

            /// Initialization token
            #[derive(Copy, Clone, Debug)]
            pub struct InitToken(());

            impl InitToken {
                /// Initialize token, performing CPU feature detection.
                pub fn init() -> Self {
                    init()
                }

                /// Initialize token and return a `bool` indicating if the feature is supported.
                pub fn init_get() -> (Self, bool) {
                    init_get()
                }

                /// Get initialized value.
                #[inline(always)]
                pub fn get(&self) -> bool {
                    $crate::__unless_target_features! {
                        $($tf),+ => {
                            STORAGE.load(Relaxed) == 1
                        }
                    }
                }
            }

            /// Get stored value and initialization token,
            /// initializing underlying storage if needed.
            #[inline]
            pub fn init_get() -> (InitToken, bool) {
                let res = $crate::__unless_target_features! {
                    $($tf),+ => {
                        #[cold]
                        fn init_inner() -> bool {
                            let res = $crate::__detect_target_features!($($tf),+);
                            STORAGE.store(res as u8, Relaxed);
                            res
                        }

                        // Relaxed ordering is fine, as we only have a single atomic variable.
                        let val = STORAGE.load(Relaxed);

                        if val == UNINIT {
                            init_inner()
                        } else {
                            val == 1
                        }
                    }
                };

                (InitToken(()), res)
            }

            /// Initialize underlying storage if needed and get initialization token.
            #[inline]
            pub fn init() -> InitToken {
                init_get().0
            }

            /// Initialize underlying storage if needed and get stored value.
            #[inline]
            pub fn get() -> bool {
                init_get().1
            }
        }
    };
    // The first set is matched separately from the rest only because `STATICALLY_DETECTED`
    // below needs its target features on their own. The fallback entry is introduced by `_`
    // rather than by a bare variant name because `$:ident` does not match `_`; were it a bare
    // name, the matcher could not tell it apart from one more set and would reject the whole
    // invocation with a local ambiguity error.
    (
        $mod_name:ident;
        $first_variant:ident: $($first_tf:tt),+;
        $($variant:ident: $($tf:tt),+;)*
        _ => $fallback:ident $(;)?
    ) => {
        mod $mod_name {
            use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

            /// Target feature set detected at runtime.
            ///
            /// Variants are ordered as declared, i.e. the detected one is always the first
            /// variant whose target features are all available.
            #[derive(Copy, Clone, Debug, Eq, PartialEq)]
            #[repr(u8)]
            pub enum Features {
                #[doc = concat!("Available target features:", $(" `", $first_tf, "`",)+)]
                $first_variant,
                $(
                    #[doc = concat!("Available target features:", $(" `", $tf, "`",)+)]
                    $variant,
                )*
                /// None of the declared target feature sets is available.
                $fallback,
            }

            // Value stored in `STORAGE` until CPU feature detection has been performed.
            //
            // `Features` is `#[repr(u8)]` and does not use explicit discriminants, so its
            // tags are exactly `0..=$fallback` and the value right past the last variant can
            // not collide with any of them.
            const UNINIT: u8 = Features::$fallback as u8 + 1;

            // Every `Features` tag has to stay below `UNINIT`, otherwise `init_get` could not
            // tell the uninitialized state apart and the transmutes below would be unsound.
            const _: () = {
                assert!((Features::$first_variant as u8) < UNINIT);
                $(assert!((Features::$variant as u8) < UNINIT);)*
                assert!((Features::$fallback as u8) < UNINIT);
            };

            // Set when all target features of the first declared set are enabled at compile
            // time. That set is probed first, so it is then always the detected one and no
            // runtime detection is necessary.
            const STATICALLY_DETECTED: bool = cfg!(all($(target_feature = $first_tf,)+));

            static STORAGE: AtomicU8 = AtomicU8::new(UNINIT);

            /// Initialization token
            #[derive(Copy, Clone, Debug)]
            pub struct InitToken(());

            impl InitToken {
                /// Initialize token, performing CPU feature detection.
                pub fn init() -> Self {
                    init()
                }

                /// Initialize token and return the detected target feature set.
                pub fn init_get() -> (Self, Features) {
                    init_get()
                }

                /// Get initialized value.
                #[inline(always)]
                pub fn get(&self) -> Features {
                    if STATICALLY_DETECTED {
                        Features::$first_variant
                    } else {
                        let val = STORAGE.load(Relaxed);

                        // SAFETY: `InitToken` can only be obtained from `init_get`, which
                        // stores the tag of a valid `Features` value into `STORAGE` before
                        // constructing the token, and the tag is never modified afterwards.
                        unsafe { core::mem::transmute::<u8, Features>(val) }
                    }
                }
            }

            #[cold]
            fn init_inner() -> Features {
                let res = 'detect: {
                    if $crate::__unless_target_features! {
                        $($first_tf),+ => { $crate::__detect_target_features!($($first_tf),+) }
                    } {
                        break 'detect Features::$first_variant;
                    }

                    $(
                        if $crate::__unless_target_features! {
                            $($tf),+ => { $crate::__detect_target_features!($($tf),+) }
                        } {
                            break 'detect Features::$variant;
                        }
                    )*

                    Features::$fallback
                };

                STORAGE.store(res as u8, Relaxed);

                res
            }

            /// Get detected target feature set and initialization token,
            /// initializing underlying storage if needed.
            #[inline]
            pub fn init_get() -> (InitToken, Features) {
                let res = if STATICALLY_DETECTED {
                    Features::$first_variant
                } else {
                    // Relaxed ordering is fine, as we only have a single atomic variable.
                    let val = STORAGE.load(Relaxed);

                    if val == UNINIT {
                        init_inner()
                    } else {
                        // SAFETY: `STORAGE` contains either `UNINIT`, which is handled
                        // above, or the tag of a valid `Features` value written by
                        // `init_inner`.
                        unsafe { core::mem::transmute::<u8, Features>(val) }
                    }
                };

                (InitToken(()), res)
            }

            /// Initialize underlying storage if needed and get initialization token.
            #[inline]
            pub fn init() -> InitToken {
                init_get().0
            }

            /// Initialize underlying storage if needed and get detected target feature set.
            #[inline]
            pub fn get() -> Features {
                init_get().1
            }
        }
    };
    ($mod_name:ident; _ => $fallback:ident $(;)?) => {
        compile_error!("`cpufeatures::new!` expects at least one target feature set");
    };
    ($mod_name:ident; $($variant:ident: $($tf:tt),+;)+) => {
        compile_error!(
            "`cpufeatures::new!` expects a trailing `_ => <variant>;` entry naming the \
             variant used when none of the target feature sets is available"
        );
    };
}
