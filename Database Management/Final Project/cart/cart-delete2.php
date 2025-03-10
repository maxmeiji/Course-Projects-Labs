<?php
$pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8', 'root', ''); //連結資料庫的動作，root資料庫裡的預設帳號名稱，空值為密碼 預設是null

$sql=$pdo->prepare('DELETE FROM cart WHERE thing_id=?');
$sql->execute([$_REQUEST['thing_id']]);
?>