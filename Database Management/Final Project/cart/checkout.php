<?php
session_start();
@$id=$_REQUEST['c_id'];
foreach ($_SESSION['commodity'] as $id=>$commodity) { 
@$count=$commodity['count'];    
$sql = "UPDATE commodity SET quantity = quantity - '$count' WHERE c_id='$id'";}
require 'clearcart.php';
?>
