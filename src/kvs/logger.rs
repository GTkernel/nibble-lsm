use log;
use log::{LogRecord, LogLevel, LogMetadata};
use log::{SetLoggerError, LogLevelFilter};
use std::thread;
use std::env;
use std::sync::{Once, ONCE_INIT};

use common::*;
use clock;

pub struct SimpleLogger {
    level: LogLevel,
}

/// A simple logger. Functions invoked by log crate
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &LogMetadata) -> bool {
        metadata.level() <= self.level
    }
    fn log(&self, record: &LogRecord) {
        if self.enabled(record.metadata()) {
            let tid = get_tid();
            let tname = match thread::current().name() {
                None => String::from("_noname_"),
                Some(n) => String::from(n),
            };
            let loc = record.location();
            let module = loc.module_path();
            let file = match loc.file().rfind("/") {
                None => loc.file(),
                Some(idx) => loc.file().split_at(idx+1).1,
            };
            println!("<{} {}-{}> {}/{}:{:4} [{}] {}",
                     clock::now(), tid, tname, module, file,
                     loc.line(), record.level(), record.args());
        }
    }
}

impl SimpleLogger {
    /// Initialize the logger
    pub fn init(level: LogLevel) -> Result<(), SetLoggerError> {
        log::set_logger(|max_log_level| {
            let l = match level {
                LogLevel::Error => LogLevelFilter::Error,
                LogLevel::Warn => LogLevelFilter::Warn,
                LogLevel::Info => LogLevelFilter::Info,
                LogLevel::Debug => LogLevelFilter::Debug,
                LogLevel::Trace => LogLevelFilter::Trace,
            };
            max_log_level.set(l);
            Box::new(SimpleLogger { level: level } )
        })
    }
}

/// Use to toggle debug messages without recompiling.
/// 1: Error 2: Warn 3: Info 4: Debug 5: Trace
const LOGGER_ENV: &'static str = "NIBDEBUG";

/// Ensure we init logger only once.
static START: Once = ONCE_INIT;

/// Call this to enable logging.
pub fn enable() {
    START.call_once( || {
        if let Some(osstr) = env::var_os(LOGGER_ENV) {
            let s = osstr.to_str().unwrap();
            let v = match u64::from_str_radix(s, 10) {
                Err(e) => panic!("{}: {:?}", LOGGER_ENV, e),
                Ok(x) => x,
            };
            let level = match v {
                1 => LogLevel::Error,
                2 => LogLevel::Warn,
                3 => LogLevel::Info,
                4 => LogLevel::Debug,
                5 => LogLevel::Trace,
                _ => panic!("{}: must be in range [1,5]", LOGGER_ENV),
            };
            let _ = SimpleLogger::init(level);
            info!("{} cycles per second",
                  clock::per_second());
        }
    });
}
