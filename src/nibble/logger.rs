use log;
use log::{LogRecord, LogLevel, LogMetadata};
use log::{SetLoggerError, LogLevelFilter};

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
            println!("{}/{}:{} [{}] {}",
                     record.location().module_path(),
                     record.location().file(), record.location().line(),
                     record.level(), record.args());
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

/// Invoked by unit tests to enable logging.
#[cfg(test)]
pub fn enable() {
    let _ = SimpleLogger::init(LogLevel::Debug);
}
