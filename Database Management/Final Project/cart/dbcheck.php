<?php
function dbcheck() {
  $servername = "localhost";
  $username = "root";
  $password = "";
  $dbname="final"; //自己已經架設好資料庫!!
  $conn = new mysqli($servername, $username, $password);
  if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
  }

  return  $conn = new mysqli($servername, $username, $password, $dbname);
}
?>