-- Create the RapRank Database
CREATE DATABASE raprank_db;

-- Create the RapRank Database User
CREATE USER raprank_user WITH PASSWORD 'raprank123';

-- Grant Privileges
GRANT ALL PRIVILEGES ON DATABASE raprank_db TO raprank_user;

-- Reset the superuser postgres password to 'postgres'
ALTER USER postgres WITH PASSWORD 'postgres';
