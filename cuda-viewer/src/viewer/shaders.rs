use std::{path::{Path, PathBuf}, fmt::Display};

use ash::{vk, prelude::VkResult, Device};
use colored::Colorize;
use shaderc::{Compiler, CompilationArtifact};

#[derive(Debug)]
pub struct CompilationWarnings {
    pub count: usize,
    pub string: String
}

impl From<&CompilationArtifact> for CompilationWarnings {
    fn from(artifact: &CompilationArtifact) -> Self {
        Self {
            count: artifact.get_num_warnings() as _,
            string: artifact.get_warning_messages()
        }
    }
}

impl Display for CompilationWarnings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = if self.count > 0 {
            format!(": {}", self.string)
        } else {
            "".to_string()
        };
        
        write!(f, "{} warning(s){}", self.count, s.yellow())
    }
}

pub struct ShaderCompiler {
    compiler: Compiler
}

type CompiledBinary = (Box<[u32]>, CompilationWarnings);

impl ShaderCompiler {
    pub fn new() -> Option<Self> {
        let compiler = Compiler::new()?;

        Some(Self {
            compiler
        })
    }
    
    pub fn compile<P: AsRef<Path>>(
        &self,
        filename: P,
        kind: shaderc::ShaderKind,
        entry_point_name: &str
    ) -> Result<CompiledBinary, Box<dyn std::error::Error>>
    {
        let full_path = {
            let mut p = ["..", "cuda-viewer", "src", "shaders"].into_iter().collect::<PathBuf>();
            p.push(&filename);
            p
        };

        let source = std::fs::read_to_string(full_path)?;
        
        let artifact = self.compiler.compile_into_spirv(
            &source,
            kind,
            &filename.as_ref().to_string_lossy(),
            entry_point_name,
            None
        )?;

        Ok((
            artifact.as_binary().into(),
            From::from(&artifact)
        ))
    }
}

pub unsafe fn create_vk_shader_module(device: &Device, binary: &[u32]) -> VkResult<vk::ShaderModule> {
    let create_info = vk::ShaderModuleCreateInfo::builder()
        .code(binary)
        .build();

    device.create_shader_module(&create_info, None)
}
