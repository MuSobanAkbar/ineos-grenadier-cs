terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "portfolio" {
  bucket = "soban-ineos-portfolio-2026"   
  tags = {
    ManagedBy   = "terraform"
    Project     = "ineos-grandier-cs"
    Environment = "Dev"
  }
}

output "bucket_arn" {
  value = aws_s3_bucket.portfolio.arn
}
