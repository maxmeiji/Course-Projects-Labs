<?php
session_start();
unset($_SESSION['commodity']);
header("Location: /final/view/final2.php"); 
?>