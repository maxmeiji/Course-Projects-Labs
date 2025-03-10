<?php



//資料庫的位置
session_start();

$host = 'localhost';
//以root管理者帳號進入資料庫
$dbuser = 'root';
//root的資料庫密碼
$dbpw = '';
//登入後要使用的資料庫
$dbname = 'final';

$_SESSION['link'] = mysqli_connect($host, $dbuser, $dbpw, $dbname);

if ($_SESSION['link'])
{

  mysqli_query($_SESSION['link'], "SET NAMES utf8");

}
else
{

  echo '無法連線mysql資料庫 :<br/>' . mysqli_connect_error();
}



?>
