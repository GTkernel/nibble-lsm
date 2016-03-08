use log;
use log::{LogRecord, LogLevel, LogMetadata};
use log::{SetLoggerError, LogLevelFilter};

pub struct SimpleLogger;

/// A simple logger. Functions invoked by log crate
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &LogMetadata) -> bool {
        metadata.level() <= LogLevel::Info
    }
    fn log(&self, record: &LogRecord) {
        if self.enabled(record.metadata()) {
            println!("{}/{}:{} [{}] {}",
                     record.location().module_path(),
                     record.location().file(), record.location().line(),
                     record.level(), record.args());
        }
    }
}

impl SimpleLogger {
    /// Initialize the logger
    pub fn init() -> Result<(), SetLoggerError> {
        log::set_logger(|max_log_level| {
            max_log_level.set(LogLevelFilter::Info);
            Box::new(SimpleLogger)
        })
    }
}
